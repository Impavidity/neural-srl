# -*- coding: utf-8 -*-
from __future__ import print_function
import re
import os


def evaluate(input_file, gold):
  result = os.popen("perl bin/eval09.pl -q -b -g " + gold + " -s " + input_file).read()
  #print(result)
  start = re.search('Labeled precision:',result).span()
  #print(start)
  end = re.search('Unlabeled precision:',result).span()
  #print(end)
  srlString = result[start[0]:end[0]].strip().split()
  #print(srlString)
  p = float(srlString[12])
  r = float(srlString[26])
  f = float(srlString[30])
  #print(p,r,f)
  return p,r,f

  #return float(result.strip().split()[27]), float(result.strip().split()[28]), float(result.strip().split()[29])

def SenseEvaluate(source, gold):
  fs = open(source)
  fg = open(gold)
  fsline = fs.readlines()
  fgline = fg.readlines()
  if (len(fsline) != len(fgline)):
    print("In the Sense Evalutation, The file length is not the same")
    print("Gold File",len(fgline))
    print("Source File",len(fsline))
    exit()
  total = 0
  right = 0
  sents = 0
  for i in xrange(len(fsline)):
    sline = fsline[i].strip().split()
    gline = fgline[i].strip().split()
    if sline and gline:
      if (sline[1] != gline[1]):
        print("Word Mismatch in line %d" %(i+1))
        exit()
      if gline[5] == "0" and gline[6] == "-1":
        continue
      elif gline[5] == "1" and gline[6] != "-1":
        total += 1
        if gline[6] == sline[3]:
          right += 1
      else:
        print("Error in Gold File in line"%(i+1))
        exit()
    elif not sline and not gline:
      sents += 1
    else:
      print("Mismatch in line %d" %(i+1))
      exit()
  if (total != sents):
    print("Sents number is not the same as total verb")
    exit()
  acc = float(right)/float(total)
  print("There are %d sentence." %(total))
  return acc





if __name__=="__main__":
  #evaluate("../Conll09/CoNLL2009-ST-English-development.txt", "../Conll09/CoNLL2009-ST-English-development.txt")
  #SenseEvaluate()
  pass