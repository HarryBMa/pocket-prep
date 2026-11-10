//! Downloads a file over HTTP(S), verified by size and SHA-256.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("download cancelled")]
    Cancelled,
    #[error("no download URL")]
    NoUrls,
    #[error("{url}: {source}")]
    Http { url: String, source: reqwest::Error },
    #[error("{url}: {source}")]
    Network { url: String, source: std::io::Error },
    #[error("{url}: unexpected HTTP status {status}")]
    Status { url: String, status: u16 },
    #[error("{}: {source}", path.display())]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("the downloaded file is corrupt: expected {expected}, got {actual}")]
    Corrupt { expected: String, actual: String },
    #[error("cannot convert the model: {0}")]
    Convert(String),
}

/// What the downloaded file must be.
#[derive(Clone, Copy, Debug)]
pub struct Expected<'a> {
    pub size: u64,
    /// Lower-case hex.
    pub sha256: &'a str,
}

const BUFFER: usize = 1 << 16;

/// Downloads `dest` from the first of `urls` that works.
///
/// Data goes to `<dest>.part`, renamed to `dest` only once its size and SHA-256
/// match. A `.part` left by a cancelled or failed download is resumed with an
/// HTTP range request when the server supports it. A corrupt download is deleted.
///
/// `progress` gets bytes done and the total.
pub fn download(
    urls: &[String],
    dest: &Path,
    expected: Expected,
    progress: &dyn Fn(u64, u64),
    cancel: &dyn Fn() -> bool,
) -> Result<(), DownloadError> {
    let part = part_path(dest);
    // Pocket Prep: reqwest instead of ureq, so the app has one HTTP stack. No overall timeout: a model is 700 MB.
    let agent = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .timeout(None)
        .build()
        .map_err(|source| DownloadError::Http { url: String::new(), source })?;

    let mut last_error = DownloadError::NoUrls;
    for url in urls {
        match download_from(&agent, url, &part, expected, progress, cancel) {
            Ok(()) => {
                return std::fs::rename(&part, dest).map_err(|source| DownloadError::Io {
                    path: dest.to_path_buf(),
                    source,
                });
            }
            Err(DownloadError::Cancelled) => return Err(DownloadError::Cancelled),
            Err(e) => {
                log::warn!("download failed, trying the next mirror: {e}");
                last_error = e;
            }
        }
    }
    Err(last_error)
}

fn download_from(
    agent: &reqwest::blocking::Client,
    url: &str,
    part: &Path,
    expected: Expected,
    progress: &dyn Fn(u64, u64),
    cancel: &dyn Fn() -> bool,
) -> Result<(), DownloadError> {
    let io_error = |source| DownloadError::Io {
        path: part.to_path_buf(),
        source,
    };
    let http_error = |source| DownloadError::Http {
        url: url.to_owned(),
        source,
    };

    // Resume a previous partial download: hash what is already there.
    let mut hasher = Sha256::new();
    let mut done = match std::fs::metadata(part) {
        Ok(meta) if meta.len() > 0 && meta.len() < expected.size => {
            hash_file(part, &mut hasher).map_err(io_error)?
        }
        _ => 0,
    };

    let mut request = agent.get(url);
    if done > 0 {
        request = request.header("Range", format!("bytes={done}-"));
    }
    let mut response = request.send().map_err(http_error)?;
    // The server dropped the range (e.g. the partial file is stale): start over.
    if response.status().as_u16() == 416 && done > 0 {
        done = 0;
        hasher = Sha256::new();
        response = agent.get(url).send().map_err(http_error)?;
    }
    let append = match response.status().as_u16() {
        206 if done > 0 => true,
        200 => {
            done = 0;
            hasher = Sha256::new();
            false
        }
        status => {
            return Err(DownloadError::Status {
                url: url.to_owned(),
                status,
            });
        }
    };
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(part)
        .map_err(io_error)?;

    let mut body = response;
    let mut buffer = vec![0u8; BUFFER];
    progress(done, expected.size);
    loop {
        if cancel() {
            return Err(DownloadError::Cancelled);
        }
        let n = match body.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(source) => {
                return Err(DownloadError::Network {
                    url: url.to_owned(),
                    source,
                });
            }
        };
        done += n as u64;
        if done > expected.size {
            break;
        }
        hasher.update(&buffer[..n]);
        file.write_all(&buffer[..n]).map_err(io_error)?;
        progress(done, expected.size);
    }
    file.flush().map_err(io_error)?;
    drop(file);

    let actual = hex(&hasher.finalize());
    if done != expected.size || actual != expected.sha256 {
        if let Err(e) = std::fs::remove_file(part) {
            log::warn!("cannot remove {}: {e}", part.display());
        }
        return Err(DownloadError::Corrupt {
            expected: format!("{} bytes, sha256 {}", expected.size, expected.sha256),
            actual: format!("{done} bytes, sha256 {actual}"),
        });
    }
    Ok(())
}

/// Feeds a file to `hasher` and returns its length.
fn hash_file(path: &Path, hasher: &mut Sha256) -> std::io::Result<u64> {
    let mut file = File::open(path)?;
    let mut buffer = vec![0u8; BUFFER];
    let mut len = 0;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            return Ok(len);
        }
        hasher.update(&buffer[..n]);
        len += n as u64;
    }
}

/// SHA-256 of a whole file, in lower-case hex.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut hasher = Sha256::new();
    hash_file(path, &mut hasher)?;
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    dest.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    use super::*;

    /// A tiny HTTP server for the tests. Paths starting with `/missing` get a
    /// 404; others get `content`, honoring `Range: bytes=N-` when `ranges` is
    /// set. Records the range of each request.
    struct Server {
        base: String,
        ranges: Arc<Mutex<Vec<Option<u64>>>>,
    }

    impl Server {
        fn start(content: Vec<u8>, ranges_supported: bool) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let ranges = Arc::new(Mutex::new(Vec::new()));
            let seen = ranges.clone();
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    let Ok(mut stream) = stream else { return };
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut request_line = String::new();
                    reader.read_line(&mut request_line).unwrap();
                    let mut range = None;
                    loop {
                        let mut line = String::new();
                        reader.read_line(&mut line).unwrap();
                        let line = line.trim();
                        if line.is_empty() {
                            break;
                        }
                        if let Some(value) = line.to_ascii_lowercase().strip_prefix("range: bytes=")
                        {
                            range = value.trim_end_matches('-').parse::<u64>().ok();
                        }
                    }
                    seen.lock().unwrap().push(range);
                    let path = request_line.split(' ').nth(1).unwrap_or("/");
                    let (status, body) = if path.starts_with("/missing") {
                        ("404 Not Found", &content[..0])
                    } else {
                        match range {
                            Some(start) if ranges_supported => {
                                ("206 Partial Content", &content[start as usize..])
                            }
                            _ => ("200 OK", &content[..]),
                        }
                    };
                    let header = format!(
                        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    stream.write_all(header.as_bytes()).unwrap();
                    stream.write_all(body).unwrap();
                }
            });
            Self { base, ranges }
        }

        fn url(&self, path: &str) -> String {
            format!("{}{path}", self.base)
        }

        fn ranges(&self) -> Vec<Option<u64>> {
            self.ranges.lock().unwrap().clone()
        }
    }

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("watershed-download-{name}-{}", std::process::id()));
            std::fs::remove_dir_all(&dir).ok();
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    fn content() -> Vec<u8> {
        (0..200_000u32).map(|i| (i * 7 % 251) as u8).collect()
    }

    fn sha256(bytes: &[u8]) -> String {
        hex(&Sha256::digest(bytes))
    }

    #[test]
    fn downloads_and_verifies() {
        let content = content();
        let server = Server::start(content.clone(), true);
        let dir = TestDir::new("ok");
        let dest = dir.0.join("model.bin");
        let hash = sha256(&content);
        let last = Mutex::new((0, 0));
        download(
            &[server.url("/model.bin")],
            &dest,
            Expected {
                size: content.len() as u64,
                sha256: &hash,
            },
            &|done, total| *last.lock().unwrap() = (done, total),
            &|| false,
        )
        .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), content);
        assert!(!part_path(&dest).exists());
        let size = content.len() as u64;
        assert_eq!(*last.lock().unwrap(), (size, size));
        assert_eq!(sha256_file(&dest).unwrap(), hash);
    }

    #[test]
    fn resumes_a_partial_download() {
        let content = content();
        let server = Server::start(content.clone(), true);
        let dir = TestDir::new("resume");
        let dest = dir.0.join("model.bin");
        std::fs::write(part_path(&dest), &content[..50_000]).unwrap();
        download(
            &[server.url("/model.bin")],
            &dest,
            Expected {
                size: content.len() as u64,
                sha256: &sha256(&content),
            },
            &|_, _| {},
            &|| false,
        )
        .unwrap();
        assert_eq!(server.ranges(), [Some(50_000)]);
        assert_eq!(std::fs::read(&dest).unwrap(), content);
    }

    #[test]
    fn starts_over_when_the_server_ignores_ranges() {
        let content = content();
        let server = Server::start(content.clone(), false);
        let dir = TestDir::new("norange");
        let dest = dir.0.join("model.bin");
        std::fs::write(part_path(&dest), &content[..50_000]).unwrap();
        download(
            &[server.url("/model.bin")],
            &dest,
            Expected {
                size: content.len() as u64,
                sha256: &sha256(&content),
            },
            &|_, _| {},
            &|| false,
        )
        .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), content);
    }

    #[test]
    fn falls_back_to_the_next_mirror() {
        let content = content();
        let server = Server::start(content.clone(), true);
        let dir = TestDir::new("mirror");
        let dest = dir.0.join("model.bin");
        download(
            &[server.url("/missing/model.bin"), server.url("/model.bin")],
            &dest,
            Expected {
                size: content.len() as u64,
                sha256: &sha256(&content),
            },
            &|_, _| {},
            &|| false,
        )
        .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), content);
    }

    #[test]
    fn rejects_and_deletes_a_corrupt_download() {
        let content = content();
        let server = Server::start(content.clone(), true);
        let dir = TestDir::new("corrupt");
        let dest = dir.0.join("model.bin");
        let result = download(
            &[server.url("/model.bin")],
            &dest,
            Expected {
                size: content.len() as u64,
                sha256: &sha256(b"something else"),
            },
            &|_, _| {},
            &|| false,
        );
        assert!(
            matches!(result, Err(DownloadError::Corrupt { .. })),
            "{result:?}"
        );
        assert!(!dest.exists());
        assert!(!part_path(&dest).exists());
    }

    #[test]
    fn cancelling_keeps_the_partial_file_for_later() {
        let content = content();
        let server = Server::start(content.clone(), true);
        let dir = TestDir::new("cancel");
        let dest = dir.0.join("model.bin");
        let polls = Mutex::new(0);
        let result = download(
            &[server.url("/model.bin")],
            &dest,
            Expected {
                size: content.len() as u64,
                sha256: &sha256(&content),
            },
            &|_, _| {},
            &|| {
                let mut polls = polls.lock().unwrap();
                *polls += 1;
                *polls > 1
            },
        );
        assert!(
            matches!(result, Err(DownloadError::Cancelled)),
            "{result:?}"
        );
        assert!(!dest.exists());
        assert!(part_path(&dest).exists());
    }
}
