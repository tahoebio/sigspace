library(arrow)
library(dplyr)
library(stringr)
library(fgsea)
library(furrr)
library(purrr)
library(tibble)
library(tidyr)

path_MS.ch <- "/home/ubuntu/sigspace/data/MSigDB_v2023.1/"
setwd(path_MS.ch)

gs_list <- c(
  gmtPathways("h.all.v2023.1.Hs.symbols.gmt"),
  gmtPathways("c5.go.v2023.1.Hs.symbols.gmt"),
  gmtPathways("c6.all.v2023.1.Hs.symbols.gmt"),
  gmtPathways("c2.cp.v2023.1.Hs.symbols.gmt")
)

gs_list_filtered <- gs_list %>%
  map(unique) %>%
  keep(~ length(.) >= 15 && length(.) <= 500)

run_fgsea_on_file <- function(parquet_path) {
  cell_line <- str_extract(basename(parquet_path), "c_\\d+")

  df <- read_parquet(parquet_path)

  df <- df %>%
    filter(!is.na(stat), !is.na(gene_name), !is.na(drugname_drugconc), !is.na(plate))

  df <- df %>%
    filter(n_cells_trt >= 50, n_cells_ctrl >= 50)

  df <- df %>%
    mutate(group_id = paste0(drugname_drugconc, "_", plate))

  unique_combos <- unique(df$group_id)

  results <- map_dfr(unique_combos, function(group) {
    print(group)
    df_sub <- df %>%
      filter(group_id == group)

    stat_vec <- df_sub$stat
    names(stat_vec) <- df_sub$gene_name

    stat_vec <- stat_vec[!is.na(stat_vec)]

    gene_sets <- gs_list_filtered %>%
      map(~ intersect(., names(stat_vec))) %>%
      keep(~ length(.) >= 15)

    if (length(gene_sets) == 0) {
      return(NULL)
    }

    fgsea_res <- fgsea(
      pathways = gene_sets,
      stats = stat_vec,
      minSize = 15,
      maxSize = 500
    )

    fgsea_res$cell_line <- cell_line
    fgsea_res$group_id <- group
    fgsea_res
  })

  out_path <- file.path("/home/ubuntu/sigspace/data/gsea/", paste0(cell_line, "_fgsea.parquet"))
  write_parquet(results, out_path)

  return(paste("completed:", cell_line))
}

plan(multicore, workers = parallel::detectCores() - 1)

parquet_files <- list.files("/home/ubuntu/sigspace/data/de_results", pattern = "\\.parquet$", full.names = TRUE)
dir.create("/home/ubuntu/sigspace/data/gsea", recursive = TRUE, showWarnings = FALSE)

results <- future_map(parquet_files, run_fgsea_on_file, .progress = TRUE)