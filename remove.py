# -*- coding: utf-8 -*-

def removeDul(filename):
  text = [[]]
  sw = [[]]
  with open(filename) as f:
    for line_num, line in enumerate(f):
      t = line.strip().split()
      if t:
        sw[-1].append(t[1])
        text[-1].append(line)
      else:
        if len(text)>=2:
          if sw[-1] == sw[-2]:
            text.pop()
            sw.pop()
        text.append([])
        sw.append([])
    f.close()
  if not text[-1]:
    text.pop()
  with open(filename+"_rm", "w") as f:
    for sent in text:
      for word in sent:
        f.write(word)
      f.write("\n")
    f.flush()
    f.close()

if __name__=="__main__":
  removeDul("../Data/joint.conll.test")

