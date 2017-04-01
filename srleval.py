# -*- coding: utf-8 -*-
from __future__ import print_function
import re
import os

def compare_insert(input_file, gold):
  output_file = input_file + "Insert"
  output = open(output_file, "w")
  sin = open(input_file)
  gin = open(gold)
  sin = sin.readlines()
  gin = gin.readlines()
  length = len(gin)
  index = 0
  text = [[]]
  sw = [[]]
  gw = [[]]
  for line_num, line0 in enumerate(sin):
    line = line0.strip().split()
    if line:
      sw[-1].append(line[1])
      text[-1].append(line0)
    else:
      while True:
        while True:
          lineg = gin[index].strip().split()
          index += 1
          if not lineg:
            break
          gw[-1].append(lineg[1])
        if gw[-1] == sw[-1]:
          for item in text[-1]:
            output.write(item)
          output.write("\n")
          sw.append([])
          gw.append([])
          text.append([])
          break
        else:
          for word_id, word in enumerate(gw[-1]):
            tup = [str(word_id), word] + ["_"] * 12 + ["\n"]
            output.write("\t".join(tup))
          output.write("\n")
          gw.append([])
  output.flush()
  output.close()
  return output_file




def combine_predicate(input_file):
  output_file = input_file + "Ced"
  text = [[]]
  sw = [[]]
  with open(input_file) as f:
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

def evaluate(input_file, gold):
  combine_pred = combine_predicate(input_file)
  insert_pred = compare_insert(combine_pred, gold)
  result = os.popen("perl bin/eval09.pl -q -b -g " + gold + " -s " + insert_pred).read()
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
  acc = float(right)/float(total) * 100
  print("There are %d sentence." %(total))
  return acc





if __name__=="__main__":
  #print(evaluate("../Conll09_Update/SrlDevOut", "../Conll09_Update/CoNLL2009-ST-English-development.txt"))
  #SenseEvaluate()
  pass