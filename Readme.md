# How to run 
 
Run the Preprocessing using (for more info consult preprocessing.md): 
```
python preprocessing.py
```

Converting points to splines (for more info consult point_to_spline.md):
```bash
python fit_splines.py \
  --input_dir ./data \
  --output_dir ./output_splines
```

To generate splines using (for more info consult gen_splines.md):
```bash
python gen_spline.py --input output_splines  --output generated_plants  --num_plants 10
```


