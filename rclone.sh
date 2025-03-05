module load rclone/1.63.1
rclone -v copy ./ pawsey1001_rakib:noisempathy/ \
--include "log_slurm/**" \
--include "log/**" \
--include "OTHERS/**" \
--include "src/archived/**" \
--include "src/ucvme/**" \
--include "data/**" \
--include ".declare_api_key.sh" \
--include "analysis-and-visualisation.ipynb"