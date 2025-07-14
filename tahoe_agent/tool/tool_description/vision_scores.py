description = [
    {
        "description": "Analyze vision scores from h5ad files and return top 10 vision scores for a specific cell. This tool loads vision score data from h5ad files and extracts the highest scoring features for a given cell name, returning formatted results.",
        "name": "analyze_vision_scores",
        "optional_parameters": [
            {
                "default": True,
                "description": "Whether to use differential vision scores (True) or regular vision scores (False). Differential scores show changes compared to baseline.",
                "name": "use_diff_scores",
                "type": "bool",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the directory containing the h5ad files with vision scores data, or direct path to an h5ad file. Should contain files named '20250417.diff_vision_scores_pseudobulk.public.h5ad' and '20250417.vision_scores_pseudobulk.public.h5ad'.",
                "name": "data_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Name of the cell to analyze, as stored in the 'Cell_Name_Vevo' column of the h5ad file observations.",
                "name": "cell_name",
                "type": "str",
            },
        ],
    },
]
