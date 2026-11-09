$py = ".venv-export\Scripts\python.exe"
& $py export_vela.py --base 16 --frames 128 --coord --harmonics 2 3 4 --modes int8 --out build\frontier
& $py export_vela.py --base 16 --frames 128 --taps 3 --modes int8 --out build\frontier
& $py export_vela.py --base 16 --frames 128 --coord --harmonics 2 3 4 --taps 5 --modes int8 --out build\frontier
