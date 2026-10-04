# run_chat.ps1
$env:Path += ";C:\Users\parva\venv\Lib\site-packages\openvino\libs;C:\Users\parva\venv\Lib\site-packages\openvino_genai;C:\Users\parva\venv\Lib\site-packages\openvino_tokenizers\lib"
& "build\bin\Release\chat_sample.exe" "C:\Users\parva\tinyllama-ov"