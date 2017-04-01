# -*- coding:utf-8 -*-

import os
import re
def parserEval(source, gold):
  f1 = open(source)
  f2 = open(source+".formats", "w")
  for line_num, line in enumerate(f1):
    line = line.strip().split()
    if line:
      f2.write("%s\t%s\t_\t_\t_\t_\t_\t_\t%s\t_\t%s\t_\t_\t_\n" % (line[0], line[1], line[6], line[7]))
    else:
      f2.write("\n")
  f2.flush()
  f2.close()
  if not os.path.isfile(gold+".formatg"):
    f1 = open(gold)
    f2 = open(gold+".formatg", "w")
    for line_num, line in enumerate(f1):
      line = line.strip().split()
      if line:
        f2.write("%s\t%s\t_\t_\t_\t_\t_\t_\t%s\t_\t%s\t_\t_\t_\n" % (line[0], line[1], line[6], line[7]))
      else:
        f2.write("\n")
    f2.flush()
    f2.close()
  result = os.popen("perl bin/eval09.pl -q -b -g " + gold+".formatg" + " -s " + source+".formats").read()
  #print(result)
  start = re.search('Labeled   attachment score:', result).span()
  end = re.search('Label accuracy score:', result).span()
  parserString = result[start[0]:end[0]].strip().split()
  las = float(parserString[9])
  uas = float(parserString[20])
  return uas, las



if __name__=="__main__":
  parserEval("../Conll09/dep.conll.test", "../Conll09/dep.conll.test")