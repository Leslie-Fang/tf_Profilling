with torch.autograd.profiler.profile() as prof:
  result = evaluate(args, model, tokenizer, prefix=prefix)
with open("fp32.prof", "w") as prof_f:
  prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))
  prof.export_chrome_trace("fp32.json")
