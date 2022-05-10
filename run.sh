
source load.sh

srun --mem=32000M  --gres=gpu:2 --partition=gpu_shared_course --time=24:00:00 "$@"
