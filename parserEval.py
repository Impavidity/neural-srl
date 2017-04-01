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



if __name__=="__main__":
  parserEval("../Conll09/dep.conll.test", "../Conll09/dep.conll.test")