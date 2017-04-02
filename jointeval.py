# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import re

def concate(source_file):
  output_file = source_file + "_concat"
  text = [[]]
  sw = [[]]
  with open(source_file) as f:
    for line_num, line in enumerate(f):
      t = line.strip().split()
      if t:
        text[-1].append(t)
        sw[-1].append(t[1])
      else:
        if len(sw) >= 2:
          if sw[-2] == sw[-1]:
            sw.pop()
            sup = text.pop()
            for item_num, item in enumerate(sup):
              if item[12] == 'Y':
                text[-1][item_num][12] = 'Y'
                text[-1][item_num][13] = item[13]
              text[-1][item_num].append(item[14])
        text.append([])
        sw.append([])
    f.flush()
    f.close()
  if not text[-1]:
    text.pop()
  with open(output_file, "w") as f:
    for sent in text:
      for word in sent:
        tup = "\t".join(word) + "\n"
        f.write(tup)
      f.write("\n")
    f.flush()
    f.close()


  return output_file

def jointEval(source_file, gold_file):
  fout = open(source_file+"_result_report", "w")
  source_file = concate(source_file)
  lmp = 0
  lmr = 0
  macro = 0
  result = os.popen("perl bin/eval09.pl -q -b -g " + gold_file + " -s " + source_file).read()
  start = re.search('Labeled precision:', result).span()
  end = re.search('Unlabeled precision:', result).span()
  srlString = result[start[0]:end[0]].strip().split()
  p = float(srlString[12])
  r = float(srlString[26])
  f = float(srlString[30])
  start = re.search('Labeled   attachment score:', result).span()
  end = re.search('Label accuracy score:', result).span()
  parserString = result[start[0]:end[0]].strip().split()
  las = float(parserString[9])
  uas = float(parserString[20])
  start = re.search('Labeled macro precision:', result).span()
  end = re.search('Unlabeled macro precision:', result).span()
  jointString = result[start[0]:end[0]].strip().split()
  lmp = float(jointString[3])
  lmr = float(jointString[8])
  macro = float(jointString[13])
  fout.write(result)
  fout.flush()
  fout.close()
  return uas, las, p, r, f ,lmp, lmr, macro


if __name__=="__main__":
  print(jointEval('../Conll09_Update/CoNLL2009-ST-evaluation-English.txt','../Conll09_Update/CoNLL2009-ST-evaluation-English.txt'))