finalconfig1 = {
    "modelid": "6bb167c7-fd88-4fc1-8cc9-5005b463a6b4",
    "modelname": "Decision Tree Clasifier (Gini)",
    "model_type": "n-classifier",
    "data": {
        "stage": "finalconfig:GET",
        "data": {
            "filepath":
                "6921397ab87f44d9856bf1c3d855ca76/1610755392___loan-xaa.csv.zip",
            "filename":
                "loan-xaa.csv.zip",
            "rows":
                19999,
            "features":
                111,
            "includedFeatures":
                51,
            "target_id":
                10,
            "target_name":
                "sub_grade",
            "sampleUsing":
                "Random",
            "splitUsing":
                "Cross Validation",
            "optimizeUsing":
                "Log Loss",
            "downsampling":
                False,
            "cv": {
                "folds": 5,
                "holdout": 20
            },
            "model_type":
                "n-classifier",
            "model_type_name":
                "Multi-class Calssification",
            "cols": [{
                "id": 1,
                "include": False,
                "name": "id",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 2,
                "include": False,
                "name": "member_id",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 3,
                "include": True,
                "name": "loan_amnt",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 4,
                "include": True,
                "name": "funded_amnt",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 5,
                "include": True,
                "name": "funded_amnt_inv",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 6,
                "include": True,
                "name": "term",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 7,
                "include": True,
                "name": "int_rate",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 8,
                "include": True,
                "name": "installment",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 9,
                "include": False,
                "name": "grade",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 10,
                "include": True,
                "name": "sub_grade",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 12,
                "include": True,
                "name": "emp_length",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 13,
                "include": True,
                "name": "home_ownership",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 14,
                "include": True,
                "name": "annual_inc",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 15,
                "include": True,
                "name": "verification_status",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 16,
                "include": True,
                "name": "issue_d",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 17,
                "include": True,
                "name": "loan_status",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 18,
                "include": True,
                "name": "pymnt_plan",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 21,
                "include": True,
                "name": "purpose",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 23,
                "include": True,
                "name": "zip_code",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 24,
                "include": True,
                "name": "addr_state",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 25,
                "include": True,
                "name": "dti",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 26,
                "include": True,
                "name": "delinq_2yrs",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 27,
                "include": True,
                "name": "earliest_cr_line",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 28,
                "include": True,
                "name": "fico_range_low",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 29,
                "include": True,
                "name": "fico_range_high",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 30,
                "include": True,
                "name": "inq_last_6mths",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 31,
                "include": False,
                "name": "mths_since_last_delinq",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 32,
                "include": False,
                "name": "mths_since_last_record",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 33,
                "include": True,
                "name": "open_acc",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 34,
                "include": True,
                "name": "pub_rec",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 35,
                "include": True,
                "name": "revol_bal",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 36,
                "include": True,
                "name": "revol_util",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 37,
                "include": True,
                "name": "total_acc",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 38,
                "include": True,
                "name": "initial_list_status",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 39,
                "include": True,
                "name": "out_prncp",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 40,
                "include": True,
                "name": "out_prncp_inv",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 41,
                "include": True,
                "name": "total_pymnt",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 42,
                "include": True,
                "name": "total_pymnt_inv",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 43,
                "include": True,
                "name": "total_rec_prncp",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 44,
                "include": True,
                "name": "total_rec_int",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 45,
                "include": True,
                "name": "total_rec_late_fee",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 46,
                "include": True,
                "name": "recoveries",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 47,
                "include": True,
                "name": "collection_recovery_fee",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 48,
                "include": True,
                "name": "last_pymnt_d",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 49,
                "include": True,
                "name": "last_pymnt_amnt",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 50,
                "include": False,
                "name": "next_pymnt_d",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 51,
                "include": True,
                "name": "last_credit_pull_d",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 52,
                "include": True,
                "name": "last_fico_range_high",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 53,
                "include": True,
                "name": "last_fico_range_low",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 54,
                "include": True,
                "name": "collections_12_mths_ex_med",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 55,
                "include": False,
                "name": "mths_since_last_major_derog",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 56,
                "include": True,
                "name": "policy_code",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 57,
                "include": True,
                "name": "application_type",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 58,
                "include": False,
                "name": "annual_inc_joint",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 59,
                "include": False,
                "name": "dti_joint",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 60,
                "include": False,
                "name": "verification_status_joint",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 61,
                "include": True,
                "name": "acc_now_delinq",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 62,
                "include": False,
                "name": "tot_coll_amt",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 63,
                "include": False,
                "name": "tot_cur_bal",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 64,
                "include": False,
                "name": "open_acc_6m",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 65,
                "include": False,
                "name": "open_il_6m",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 66,
                "include": False,
                "name": "open_il_12m",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 67,
                "include": False,
                "name": "open_il_24m",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 68,
                "include": False,
                "name": "mths_since_rcnt_il",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 69,
                "include": False,
                "name": "total_bal_il",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 70,
                "include": False,
                "name": "il_util",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 71,
                "include": False,
                "name": "open_rv_12m",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 72,
                "include": False,
                "name": "open_rv_24m",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 73,
                "include": False,
                "name": "max_bal_bc",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 74,
                "include": False,
                "name": "all_util",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 75,
                "include": False,
                "name": "total_rev_hi_lim",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 76,
                "include": False,
                "name": "inq_fi",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 77,
                "include": False,
                "name": "total_cu_tl",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 78,
                "include": False,
                "name": "inq_last_12m",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 79,
                "include": False,
                "name": "acc_open_past_24mths",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 80,
                "include": False,
                "name": "avg_cur_bal",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 81,
                "include": False,
                "name": "bc_open_to_buy",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 82,
                "include": False,
                "name": "bc_util",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 83,
                "include": True,
                "name": "chargeoff_within_12_mths",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 84,
                "include": True,
                "name": "delinq_amnt",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 85,
                "include": False,
                "name": "mo_sin_old_il_acct",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 86,
                "include": False,
                "name": "mo_sin_old_rev_tl_op",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 87,
                "include": False,
                "name": "mo_sin_rcnt_rev_tl_op",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 88,
                "include": False,
                "name": "mo_sin_rcnt_tl",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 89,
                "include": False,
                "name": "mort_acc",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 90,
                "include": False,
                "name": "mths_since_recent_bc",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 91,
                "include": False,
                "name": "mths_since_recent_bc_dlq",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 92,
                "include": False,
                "name": "mths_since_recent_inq",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 93,
                "include": False,
                "name": "mths_since_recent_revol_delinq",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 94,
                "include": False,
                "name": "num_accts_ever_120_pd",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 95,
                "include": False,
                "name": "num_actv_bc_tl",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 96,
                "include": False,
                "name": "num_actv_rev_tl",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 97,
                "include": False,
                "name": "num_bc_sats",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 98,
                "include": False,
                "name": "num_bc_tl",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 99,
                "include": False,
                "name": "num_il_tl",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 100,
                "include": False,
                "name": "num_op_rev_tl",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 101,
                "include": False,
                "name": "num_rev_accts",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 102,
                "include": False,
                "name": "num_rev_tl_bal_gt_0",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 103,
                "include": False,
                "name": "num_sats",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 104,
                "include": False,
                "name": "num_tl_120dpd_2m",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 105,
                "include": False,
                "name": "num_tl_30dpd",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 106,
                "include": False,
                "name": "num_tl_90g_dpd_24m",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 107,
                "include": False,
                "name": "num_tl_op_past_12m",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 108,
                "include": False,
                "name": "pct_tl_nvr_dlq",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 109,
                "include": False,
                "name": "percent_bc_gt_75",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 110,
                "include": True,
                "name": "pub_rec_bankruptcies",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 111,
                "include": True,
                "name": "tax_liens",
                "origin": "native",
                "type": "Category",
                "weight": 1,
                "imputable": False
            }, {
                "id": 112,
                "include": False,
                "name": "tot_hi_cred_lim",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 113,
                "include": False,
                "name": "total_bal_ex_mort",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 114,
                "include": False,
                "name": "total_bc_limit",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }, {
                "id": 115,
                "include": False,
                "name": "total_il_high_credit_limit",
                "origin": "native",
                "type": "Number",
                "weight": 1,
                "imputable": False
            }]
        }
    }
}
