# -*- coding:utf-8 -*-

import os
import re
def parserEval(source, gold):
  result = os.popen("perl bin/eval09.pl -q -b -g " + gold + " -s " + source).read()
  #print(result)
  start = re.search('Labeled   attachment score:', result).span()
  end = re.search('Label accuracy score:', result).span()
  parserString = result[start[0]:end[0]].strip().split()
  las = float(parserString[9])
  uas = float(parserString[20])
  return uas, las

def embedding_output(save, filename, sents, hidden_repre):
  with open(save+"/parsing_bed+filename",'a') as f:
    for sent, hidden in zip(sents, hidden_repre):
      for word, hid in zip(sent, hidden[1:]):
        f.write(word+" ")
        for item in hid:
          f.write(str(item)+" ")
        f.write("\n")



if __name__=="__main__":
  parserEval("../Conll09/dep.conll.test", "../Conll09/dep.conll.test")