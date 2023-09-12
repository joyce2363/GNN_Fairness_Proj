
# import argparse
# from train_fairGNN import call_fairGNN
# from b_training1 import call_BIND_training1
# import torch

# def main(args):
#     if args.model == 'fairGNN': 
#         call_fairGNN()
#     elif args.model == 'BIND': 
#         print("END")
#         call_BIND_training1() 

# if __name__ == '__main__':
#     print('entered MAIN')
#     # Training settings
#     parser = argparse.ArgumentParser()
#     # the only think you had to do is replace the fault with the model you want to run
#     parser.add_argument('--model',type=str,default='fairGNN',help='Model')
#     args = parser.parse_args()
#     main(args)