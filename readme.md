Command examples:

```bash
python main.py --first --train --out 101552plm --dataset AAPL --iter 7 --lr 0.3
python test.py --all --inn 101552plm --dataset AAPL
python test.py --val --inn 101552plm --dataset AAPL
python test.py --test --inn 101552plm --dataset AAPL
```

Command can't control the kernel choice yet. If you want to try different kernels, just modify Line 16 in main.py
