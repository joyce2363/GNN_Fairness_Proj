







def test(model,st,end):
        # model.eval()
        # torch.cuda.empty_cache()
        # with torch.no_grad():
            if args.model == "FairGNN":
                output, out_AUC = model(features[split_idx['test']], A_test, A_di_test, A_di_t_test, test_data, device, st, end)
            elif args.model == "BIND": 
                output, out_AUC = model(features[split_idx['test']], A_test, A_di_test, A_di_t_test, test_data, device, st, end)
