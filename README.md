# AVAI

Need rest of datset:
    Train:
        DIV2K_train_LR_bicubic_X2
        DIV2K_train_LR_bicubic_X3
        DIV2K_train_LR_bicubic_X4
        DIV2K_train_LR_unknown_X2
        DIV2K_train_LR_unknown_X3
        DIV2K_train_LR_unknown_X4
    Valid:
        DIV2K_valid_LR_bicubic_X2
        DIV2K_valid_LR_bicubic_X3
        DIV2K_valid_LR_bicubic_X4
        DIV2K_valid_LR_unknown_X2
        DIV2K_valid_LR_unknown_X3
        DIV2K_valid_LR_unknown_X4


TODO:
- Check best loss function to use (Trainer.py)
- Find out best value for input_dim (train_DIV2k.py)







Generator Class:
- Implements SRResNet (Super-Resolutiom Residual Network)
- Designed to take small image and exapnd it (filling in details)
- Residual Block is to implement residual connects allowing for it to just learn the difference/error compared to who,e image at that layer