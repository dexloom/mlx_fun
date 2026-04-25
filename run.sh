mlx-fun serve \
    --port 8899 \
    --model /Users/sombrax/.lmstudio/models/sombra/GLM-5.1-Q3-g32 \
    --idle-timeout 7200 \
    --default-top-k 100 \
    --default-repetition-penalty 1.1 \
    --enable-counting \
    --log-level INFO 2>&1 | tee /tmp/mlx_fun_server.log
