def get_params():
    params = {
        "image_size" : 28,
        "num_labels" : 10,
        "num_channels" : 1,
        "batch_size" : 200,
        "patch_size" : 5,
        "depth" : 16,
        "num_hidden" : 64,
        "num_steps" : 50001,
        "dropout_prob" : 3,
        "l2_regularization_rate" : 0.1,
        "learning_startrate" : 0.01,
        "learning_decay" : 0.97,
        "learning_decay_steps" : 100,
        "num_log_steps" : 1000,
        "log_folder" : "conv_logs_1"
    }
    return params;
