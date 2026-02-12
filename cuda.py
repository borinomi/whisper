import torch

print("=" * 50)
print("CUDA 사용 가능:", torch.cuda.is_available())
print("PyTorch 버전:", torch.__version__)
print("CUDA 버전:", torch.version.cuda)
print("cuDNN 버전:", torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A")

if torch.cuda.is_available():
    print("GPU 개수:", torch.cuda.device_count())
    print("현재 GPU:", torch.cuda.current_device())
    print("GPU 이름:", torch.cuda.get_device_name(0))
else:
    print("\n⚠️  CUDA를 사용할 수 없습니다.")
    print("가능한 원인:")
    print("1. NVIDIA GPU가 없음")
    print("2. CUDA Toolkit이 설치되지 않음")
    print("3. GPU 드라이버가 오래됨")
    print("4. PyTorch CPU 버전이 설치됨")
print("=" * 50)