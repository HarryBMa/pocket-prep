//! The models Watershed can download (`models.toml`, embedded in the binary).

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use serde::Deserialize;

use crate::download::{self, DownloadError, Expected};
use crate::{BurnSeparator, SeparationError};

#[derive(Debug, Deserialize)]
struct CatalogFile {
    model: Vec<ModelEntry>,
}

static CATALOG: LazyLock<Vec<ModelEntry>> = LazyLock::new(|| {
    toml::from_str::<CatalogFile>(include_str!("../models.toml"))
        .expect("models.toml is checked by the tests")
        .model
});

/// One downloadable model. See `models.toml` for what each field means.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    /// The PyTorch checkpoint to download.
    pub file: String,
    pub size: u64,
    pub sha256: String,
    pub urls: Vec<String>,
    pub stems: Vec<String>,
    pub license: String,
    #[serde(default)]
    pub default: bool,
    pub architecture: Architecture,
    /// The checkpoint's `.yaml`.
    pub config: ConfigFile,
    /// What the checkpoint is converted to once downloaded, and then used instead.
    pub converted: ConvertedFile,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Architecture {
    BsRoformer,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct ConfigFile {
    pub file: String,
    pub size: u64,
    pub sha256: String,
    pub urls: Vec<String>,
}

/// The checkpoint's weights in f16, made on the user's machine. Nothing to download.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct ConvertedFile {
    pub file: String,
    pub size: u64,
    pub sha256: String,
}

/// All models, in catalog order.
pub fn models() -> &'static [ModelEntry] {
    &CATALOG
}

/// The model used for separations.
pub fn default_model() -> &'static ModelEntry {
    CATALOG
        .iter()
        .find(|m| m.default)
        .expect("models.toml has a default model")
}

impl ModelEntry {
    /// Where this model lives inside the models folder once installed: the converted
    /// file, not the downloaded checkpoint.
    pub fn path_in(&self, models_dir: &Path) -> PathBuf {
        models_dir.join(&self.converted.file)
    }

    /// Whether the model's files are there with the right size. The hash is only
    /// checked when installing: hashing 350 MB at every start would be slow.
    pub fn is_installed(&self, models_dir: &Path) -> bool {
        let has = |file: &str, size| {
            std::fs::metadata(models_dir.join(file)).is_ok_and(|m| m.len() == size)
        };
        has(&self.converted.file, self.converted.size) && has(&self.config.file, self.config.size)
    }

    /// Downloads the model into `models_dir`, unless it is already installed, and
    /// returns its path. See [`download::download`] for resuming and verification.
    /// `progress` follows the checkpoint: a config is a few KB.
    ///
    /// The checkpoint is then converted (see [`ModelEntry::converted`]) and deleted. A
    /// checkpoint already there, from an interrupted install or an older version, is
    /// checked and used instead of downloading it again.
    pub fn install(
        &self,
        models_dir: &Path,
        progress: &dyn Fn(u64, u64),
        cancel: &dyn Fn() -> bool,
    ) -> Result<PathBuf, DownloadError> {
        let path = self.path_in(models_dir);
        let checkpoint = models_dir.join(&self.file);
        if self.is_installed(models_dir) {
            if checkpoint.exists() {
                remove(&checkpoint);
            }
            return Ok(path);
        }
        std::fs::create_dir_all(models_dir).map_err(|source| DownloadError::Io {
            path: models_dir.to_path_buf(),
            source,
        })?;
        let config = &self.config;
        let config_path = models_dir.join(&config.file);
        if !std::fs::metadata(&config_path).is_ok_and(|m| m.len() == config.size) {
            download::download(
                &config.urls,
                &config_path,
                Expected {
                    size: config.size,
                    sha256: &config.sha256,
                },
                &|_, _| {},
                cancel,
            )?;
        }
        let expected = Expected {
            size: self.size,
            sha256: &self.sha256,
        };
        if !is_intact(&checkpoint, expected) {
            download::download(&self.urls, &checkpoint, expected, progress, cancel)?;
        }
        convert(&checkpoint, &path, &self.converted)?;
        remove(&checkpoint);
        Ok(path)
    }

    /// Loads the installed model as a separator with this entry's stem names.
    pub fn load(&self, models_dir: &Path) -> Result<BurnSeparator, SeparationError> {
        let Architecture::BsRoformer = self.architecture;
        let config = models_dir.join(&self.config.file);
        BurnSeparator::load(&self.path_in(models_dir), &config, self.stems.clone())
    }
}

/// Whether `path` is there with the expected size and hash. A file that is not is
/// deleted.
fn is_intact(path: &Path, expected: Expected) -> bool {
    if !std::fs::metadata(path).is_ok_and(|m| m.len() == expected.size) {
        return false;
    }
    let intact = download::sha256_file(path).is_ok_and(|hash| hash == expected.sha256);
    if !intact {
        log::warn!("{} is damaged, downloading it again", path.display());
        remove(path);
    }
    intact
}

/// Converts the checkpoint into `<dest>.part`, renamed to `dest` only once its size
/// and hash match: an interrupted conversion never leaves a file that looks complete.
fn convert(checkpoint: &Path, dest: &Path, expected: &ConvertedFile) -> Result<(), DownloadError> {
    let part = download::part_path(dest);
    let io = |source| DownloadError::Io {
        path: part.clone(),
        source,
    };
    let result = watershed_roformer::convert_to_half(checkpoint, &part)
        .map_err(|e| DownloadError::Convert(e.to_string()))
        .and_then(|()| {
            let size = std::fs::metadata(&part).map_err(io)?.len();
            let hash = download::sha256_file(&part).map_err(io)?;
            if size == expected.size && hash == expected.sha256 {
                Ok(())
            } else {
                Err(DownloadError::Corrupt {
                    expected: format!("{} bytes, sha256 {}", expected.size, expected.sha256),
                    actual: format!("{size} bytes, sha256 {hash}"),
                })
            }
        })
        .and_then(|()| std::fs::rename(&part, dest).map_err(io));
    if result.is_err() {
        remove(&part);
    }
    result
}

fn remove(path: &Path) {
    if let Err(e) = std::fs::remove_file(path)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        log::warn!("cannot remove {}: {e}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn catalog_is_valid() {
        let models = models();
        assert!(!models.is_empty());
        assert_eq!(models.iter().filter(|m| m.default).count(), 1);
        let unique =
            |values: Vec<&str>| values.iter().collect::<HashSet<_>>().len() == values.len();
        assert!(unique(models.iter().map(|m| m.id.as_str()).collect()));
        assert!(unique(models.iter().map(|m| m.file.as_str()).collect()));
        let hash = |sha256: &str| {
            sha256.len() == 64
                && sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        };
        for m in models {
            assert!(m.size > 0, "{}", m.id);
            assert!(hash(&m.sha256), "{}: sha256 must be lower-case hex", m.id);
            assert!(!m.urls.is_empty(), "{}", m.id);
            assert!(m.urls.iter().all(|u| u.starts_with("https://")), "{}", m.id);
            assert!(!m.stems.is_empty() && unique(m.stems.iter().map(String::as_str).collect()));
            // Stem names become file names.
            assert!(
                m.stems
                    .iter()
                    .all(|s| s.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')),
                "{}",
                m.id
            );
            assert!(m.config.size > 0, "{}", m.id);
            assert!(hash(&m.config.sha256), "{}", m.id);
            assert!(!m.config.urls.is_empty(), "{}", m.id);
            assert!(m.config.urls.iter().all(|u| u.starts_with("https://")));
            assert!(m.converted.size > 0, "{}", m.id);
            assert!(hash(&m.converted.sha256), "{}", m.id);
            assert!(m.converted.file.ends_with(".safetensors"), "{}", m.id);
            assert!(unique(vec![&m.file, &m.config.file, &m.converted.file]));
        }
    }

    #[test]
    fn installed_means_the_converted_file_and_the_config_at_the_right_size() {
        let dir = std::env::temp_dir().join(format!("watershed-installed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut model = default_model().clone();
        model.converted.size = 4;
        model.config.size = 2;
        // The checkpoint alone is not enough: it is converted when installing.
        std::fs::write(dir.join(&model.file), b"abcdefgh").unwrap();
        std::fs::write(dir.join(&model.config.file), b"ab").unwrap();
        assert!(!model.is_installed(&dir));
        std::fs::write(model.path_in(&dir), b"abc").unwrap();
        assert!(!model.is_installed(&dir));
        std::fs::write(model.path_in(&dir), b"abcd").unwrap();
        assert!(model.is_installed(&dir));
        std::fs::remove_file(dir.join(&model.config.file)).unwrap();
        assert!(!model.is_installed(&dir));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The tiny test model as a catalog entry whose checkpoint is already in `dir`,
    /// so that installing it downloads nothing, and the hash of its conversion.
    fn tiny_model(dir: &Path) -> ModelEntry {
        let tiny =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../roformer/tests/fixtures/tiny.ckpt");
        let mut model = default_model().clone();
        let checkpoint = dir.join(&model.file);
        std::fs::copy(&tiny, &checkpoint).unwrap();
        model.size = std::fs::metadata(&checkpoint).unwrap().len();
        model.sha256 = download::sha256_file(&checkpoint).unwrap();
        model.urls = vec!["https://localhost.invalid/never-used".into()];

        let reference = dir.join("reference.safetensors");
        watershed_roformer::convert_to_half(&checkpoint, &reference).unwrap();
        model.converted.size = std::fs::metadata(&reference).unwrap().len();
        model.converted.sha256 = download::sha256_file(&reference).unwrap();
        std::fs::remove_file(&reference).unwrap();
        model.config.size = 2;
        std::fs::write(dir.join(&model.config.file), b"ab").unwrap();
        model
    }

    #[test]
    fn installing_converts_the_checkpoint_and_deletes_it() {
        let dir = std::env::temp_dir().join(format!("watershed-convert-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = tiny_model(&dir);
        assert!(!model.is_installed(&dir));
        let path = model.install(&dir, &|_, _| {}, &|| false).unwrap();
        assert_eq!(path, model.path_in(&dir));
        assert!(model.is_installed(&dir));
        assert!(!dir.join(&model.file).exists());
        assert!(!download::part_path(&path).exists());

        // A checkpoint left over next to an installed model is removed.
        std::fs::write(dir.join(&model.file), b"left over").unwrap();
        model.install(&dir, &|_, _| {}, &|| false).unwrap();
        assert!(!dir.join(&model.file).exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_conversion_with_the_wrong_hash_leaves_nothing_that_looks_installed() {
        let dir = std::env::temp_dir().join(format!("watershed-badconv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut model = tiny_model(&dir);
        model.converted.sha256 = "0".repeat(64);
        let result = model.install(&dir, &|_, _| {}, &|| false);
        assert!(
            matches!(result, Err(DownloadError::Corrupt { .. })),
            "{result:?}"
        );
        assert!(!model.path_in(&dir).exists());
        assert!(!download::part_path(&model.path_in(&dir)).exists());
        assert!(!model.is_installed(&dir));
        // The checkpoint is kept: it was fine.
        assert!(dir.join(&model.file).exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
