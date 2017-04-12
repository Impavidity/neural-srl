# -*-coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import argparse
from scipy import stats

def eval(file_source, file_gold, sense):
  fs = open(file_source)
  fg = open(file_gold)
  fs = fs.readlines()
  fg = fg.readlines()
  per_right_sense = []
  per_total_sense = []
  per_right_argument = []
  per_gold_argument = []
  per_retrieve_argument = []
  temp_right_sense = 0
  temp_total_sense = 0
  temp_right_argument = 0
  temp_gold_argument = 0
  temp_retrieve_argument = 0

  if len(fs) != len(fg):
    print("The length is not match, please check")
    print("source %d gold %d" % (len(fs), len(fg)))
  for line_num, (lines, lineg) in enumerate(zip(fs, fg)):
    try:
      lines = lines.strip().split()
      lineg = lineg.strip().split()
      if not lines and not lineg:
        per_right_sense.append(temp_right_sense)
        per_total_sense.append(temp_total_sense)
        per_right_argument.append(temp_right_argument)
        per_gold_argument.append(temp_gold_argument)
        per_retrieve_argument.append(temp_retrieve_argument)
        temp_right_sense = 0
        temp_total_sense = 0
        temp_right_argument = 0
        temp_gold_argument = 0
        temp_retrieve_argument = 0
        continue
      elif lines is not None and lineg is not None:
        if lineg[12] == 'Y':
          temp_total_sense += 1
          if lines[12] == 'Y' and lines[13].split('.')[1] == lineg[13].split('.')[1]:
            temp_right_sense += 1
        for p in xrange(14, len(lineg)):
          if lineg[p] != '_':
            temp_gold_argument += 1
          if lines[p] != '_':
            temp_retrieve_argument += 1
          if lineg[p] != '_' and lineg[p] == lines[p]:
            temp_right_argument += 1
      else:
        print("Error state in line %d" % (line_num))
    except:
      print("Error in line %d" % (line_num))
      print(lineg)
      print(lines)
      exit()

  right_sense = sum(per_right_sense)
  total_sense = sum(per_total_sense)
  right_argument = sum(per_right_argument)
  gold_arguent = sum(per_gold_argument)
  retrieve_argument = sum(per_retrieve_argument)
  if sense:
    print("## With sense")
    print("Precision: (%d + %d) / (%d + %d) = %f" %
          (right_argument, right_sense, retrieve_argument, total_sense,
           (right_argument + right_sense) / (retrieve_argument + total_sense)))
    print("Recall: (%d + %d) / (%d + %d) = %f" %
          (right_argument, right_sense, gold_arguent, total_sense,
           (right_argument + right_sense) / (gold_arguent + total_sense)))
    p = (right_argument + right_sense) / (retrieve_argument + total_sense)
    r = (right_argument + right_sense) / (gold_arguent + total_sense)
    print("F score: %f" % (2 * p * r / (p + r)))
    print("Sense Precision: %f " % (right_sense / total_sense))
  else:
    print("## without sense")
    print("Precision: %d / %d = %f" % (right_argument, retrieve_argument,
                                       right_argument / retrieve_argument))
    print("Recall: %d / %d = %f" % (right_argument, gold_arguent,
                                    right_argument / gold_arguent))
    p = right_argument / retrieve_argument
    r = right_argument / gold_arguent
    print("F score: %f" % (2 * p * r / (p + r)))

  file_name = "_per_sent_stat"
  if sense:
    file_name += "_with_sense"
  else:
    file_name += "_without_sense"
  file_name = file_source + file_name
  fout = open(file_name, "w")
  for item_right_sense, item_total_sense, \
    item_right_argument, item_retrieve_argument, item_gold_argument \
    in zip(per_right_sense, per_total_sense, per_right_argument, \
           per_retrieve_argument, per_gold_argument):
    if sense:
      if item_retrieve_argument + item_total_sense != 0:
        p = (item_right_argument + item_right_sense) / (item_retrieve_argument + item_total_sense)
      else:
        p = 0
      if item_gold_argument + item_total_sense != 0:
        r = (item_right_argument + item_right_argument) / (item_gold_argument + item_total_sense)
      else:
        r = 0
    else:
      if item_retrieve_argument != 0:
        p = (item_right_argument) / (item_retrieve_argument)
      else:
        p = 0
      if item_gold_argument != 0:
        r = (item_right_argument) / (item_gold_argument)
      else:
        r = 0
    if (p + r) != 0:
      f = 2 * p * r / (p + r)
    else:
      f = 0
    fout.write(str(f) + "\n")


def ttest(file_old, file_new):
  fold = open(file_old)
  fnew = open(file_new)
  base = [float(num.strip()) for num in fold.readlines()]
  test = [float(num.strip()) for num in fnew.readlines()]
  if len(base) != len(test):
    print("Length Error")
    exit()
    print("base %d test %d" % (len(base), len(test)))
  a, p = stats.ttest_rel(base, test)
  print("a = %f , p = %f" % (a, p))


if __name__=="__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument('--source', help="The source file | The original file")
  argparser.add_argument('--gold', help="The gold file | The new file")
  argparser.add_argument('--sense', help="Include the sense", action="store_true")
  argparser.add_argument('--fscore', help="Evaluate the F score", action="store_true")
  argparser.add_argument('--ttest', help="T test", action="store_true")
  arg = argparser.parse_args()

  if arg.source == None or arg.gold == None:
    print("Source or Gold file is None")
  if arg.fscore:
    print(arg.sense)
    eval(arg.source, arg.gold, arg.sense)
  if arg.ttest:
    ttest(arg.source, arg.gold)



