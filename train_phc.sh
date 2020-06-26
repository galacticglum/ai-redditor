python ai-redditor\gpt2\train.py data\phc\pornhub_comments_train.data --do-train --max-checkpoints 5 \
    --model gpt2 --outdir output\phc-117M --log-frequency 100 --save-frequency 100 --epochs 10 --do-eval \
    --eval-dataset data\phc\pornhub_comments_test.data --gradient-accumulation-steps 2 --train-batch-size 1 \
    --eval-batch-size 1 --lr-warmup-steps 1000 --special-tokens data\special_tokens\phc_special_tokens.json