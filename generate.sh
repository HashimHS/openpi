data_dir=${1}
repo_id=${2}
uv run examples/kuka/convert_kuka_data_to_lerobot_robotwin.py --raw_dir $data_dir --repo_id $repo_id
