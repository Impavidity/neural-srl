#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from lib.networks import *


# TODO make the optimizer class inherit from Configurable
# TODO bayesian hyperparameter optimization
# TODO start a UD tagger/parser pipeline
#***************************************************************

if __name__ == '__main__':
  """"""

  import argparse

  argparser = argparse.ArgumentParser()
  argparser.add_argument('--test', action='store_true')
  argparser.add_argument('--validate', action='store_true')
  argparser.add_argument('--ood', action='store_true')
  argparser.add_argument('--load', action='store_true')
  argparser.add_argument('--stacking_dep', action="store_true")
  argparser.add_argument('--stacking_srl', action="store_true")
  argparser.add_argument('--stacking', action="store_true")
  argparser.add_argument('--srl_major', action="store_true")
  argparser.add_argument('--complicated_loss', action="store_true")
  # store_true : if it is declared, then set it as true
  args, extra_args = argparser.parse_known_args()
  #args: Namespace(load=False, model='Parser', pretrain=False, test=False)
  #extra_args: ["--some","xxxx"]
  cargs = {k: v for (k, v) in vars(Configurable.argparser.parse_args(extra_args)).iteritems() if v is not None}
  #Attention: You need to specify the model type in the config file and the command line at the same time and match
  if 'model_type' not in cargs:
    print("You need to specify the model_type")
    exit()
  print('*** '+cargs['model_type']+' ***')

  model = getattr(models, cargs['model_type'])

  if 'save_dir' in cargs and os.path.isdir(cargs['save_dir']) and not (args.test or args.load or args.validate or args.ood):
    raw_input('Save directory already exists. Press <Enter> to overwrite or <Ctrl-C> to exit.')
  if (args.test or args.load or args.validate or args.ood) and 'save_dir' in cargs:
    cargs['config_file'] = os.path.join(cargs['save_dir'], 'config.cfg')


  network = None

  # graph = tf.Graph()
  #with graph.as_default():
  tf.set_random_seed(1995)
  np.random.seed(1995)
  #print(tf.get_default_graph().seed)
  if cargs['model_type'] == "SimpleSrler":
    cargs.pop("model_type","")
    network = SimpleSRLNetwork(args, model, **cargs)
  elif cargs['model_type'] == "SenseDisamb":
    cargs.pop("model_type","")
    network = SenseDisambNetwork(args, model, **cargs)
  elif cargs['model_type'] == "Parser":
    cargs.pop("model_type","")
    network = ParserNetwork(args, model, **cargs)
  elif cargs['model_type'] == "MultiTask":
    cargs.pop("model_type","")
    network = MultiTaskNetwork(args, model, **cargs)
  else:
    print("Unsupported Model")
    exit()



  config_proto = tf.ConfigProto()
  config_proto.gpu_options.allow_growth = True
  with tf.Session(config=config_proto) as sess:
    sess.run(tf.global_variables_initializer())
    if not args.test and not args.validate and not args.ood:
      if args.load:
        #var_list = filter(lambda x: u'Srls' not in x.name and u'Verbs' not in x.name
        # and u'VerbSense' not in x.name and u'SRLClassifier_para' not in x.name
        # and u'SRLCOMRNN0' not in x.name and u'SRLCOMRNN1' not in x.name
        # and u'IsVerbs' not in x.name,tf.global_variables())
        # var_list = filter(lambda x : u'RadamOptimizer' not in x.name, tf.global_variables())
        saver = tf.train.Saver(name=network.restore_name)
        saver.restore(sess, tf.train.latest_checkpoint(network.restore_from, latest_filename=network.restore_name.lower()))
        network.is_load = True
        print("Load Model Successfully")
      else:
        os.system('echo Loading: >> %s/HEAD' % network.save_dir)
      network.train(sess)
    else:
      os.system('echo Testing: >> %s/HEAD' % network.save_dir)
      saver = tf.train.Saver(name=network.name)
      saver.restore(sess, tf.train.latest_checkpoint(network.save_dir, latest_filename=network.name.lower()))
      if network.model_type == "SenseDisamb":
        if args.test and not args.validate and not args.ood:
          acc = network.test(sess, validate=False, ood=False)
          print("## TESTING ##")
          print("Sense Accuracy %5.2f" % (acc))
        elif not args.test and not args.ood and args.validate:
          acc = network.test(sess, validate=True, ood=False)
          print("## VALIDATION ##")
          print("Sense Accuracy %5.2f" % (acc))
        elif not args.test and not args.validate and args.ood:
          acc = network.test(sess, validate=False, ood=True)
          print("## OOD ##")
          print("Sense Accuracy %5.2f" % (acc))
      if network.model_type == "SimpleSrler":
        if args.test and not args.validate and not args.ood:
          p, r, f = network.test(sess, validate=False, ood=False)
          print("## TESTING ##")
          print("Srler P : %5.2f R : %5.2f F : %5.2f" % (p, r, f))
        elif not args.test and not args.ood and args.validate:
          p, r, f = network.test(sess, validate=True, ood=False)
          print("## VALIDATION ##")
          print("Srler P : %5.2f R : %5.2f F : %5.2f" % (p, r, f))
        elif not args.test and not args.validate and args.ood:
          p, r, f = network.test(sess, validate=False, ood=True)
          print("## OOD ##")
          print("Srler P : %5.2f R : %5.2f F : %5.2f" % (p, r, f))
      if network.model_type == "Parser":
        if args.test and not args.validate and not args.ood:
          uas, las = network.test(sess, validate=False, ood=False)
          print("## TESTING ##")
          print("Parser UAS : %5.2f LAS : %5.2f" % (uas, las))
        elif not args.test and not args.ood and args.validate:
          uas, las = network.test(sess, validate=True, ood=False)
          print("## VALIDATION ##")
          print("Parser UAS : %5.2f LAS : %5.2f" % (uas, las))
        elif not args.test and not args.validate and args.ood:
          uas, las = network.test(sess, validate=False, ood=True)
          print("## OOD ##")
          print("Parser UAS : %5.2f LAS : %5.2f" % (uas, las))
      if network.model_type == "MultiTask":
        if args.test and not args.validate and not args.ood:
          uas, las, p, r, f, lmp, lmr, macro = network.test(sess, validate=False, ood=False) # uas, las, p, r, f, lmp, lmr, macro
          print("## TESTING ##")
          print("Srler P : %5.2f R : %5.2f F : %5.2f" % (p, r, f))
          print("Parser UAS : %5.2f LAS : %5.2f" % (uas, las))
          print("lmp : %5.2f lmr : %5.2f macro : %5.2f " % (lmp, lmr, macro))
        elif not args.test and not args.ood and args.validate:
          uas, las, p, r, f, lmp, lmr, macro = network.test(sess, validate=True, ood=False)
          print("## VALIDATION ##")
          print("Srler P : %5.2f R : %5.2f F : %5.2f" % (p, r, f))
          print("Parser UAS : %5.2f LAS : %5.2f" % (uas, las))
          print("lmp : %5.2f lmr : %5.2f macro : %5.2f " % (lmp, lmr, macro))
        elif not args.test and not args.validate and args.ood:
          uas, las, p, r, f, lmp, lmr, macro = network.test(sess, validate=False, ood=True)
          print("## OOD ##")
          print("Srler P : %5.2f R : %5.2f F : %5.2f" % (p, r, f))
          print("Parser UAS : %5.2f LAS : %5.2f" % (uas, las))
          print("lmp : %5.2f lmr : %5.2f macro : %5.2f " % (lmp, lmr, macro))
