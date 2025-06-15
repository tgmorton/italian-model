# scripts/analyze_results.R

# --- 1. SETUP ---

# Load necessary libraries
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(svglite)) # For saving plots

# --- !! USER CONFIGURATION !! ---
# Set the model size tag and directories you want to analyze here.
model_tag <- "100M" # <--- EDIT THIS (e.g., "25M", "50M")
analysis_base_dir <- "analysis"
# --------------------------------

# --- Define Paths ---
input_csv_path <- file.path(analysis_base_dir, paste0(model_tag, "_surprisal_analysis.csv"))

# Create output directories for the plots
line_plot_dir <- file.path(analysis_base_dir, model_tag, "surprisal_over_time")
forest_plot_dir <- file.path(analysis_base_dir, model_tag, "final_checkpoint_effects")
dir.create(line_plot_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(forest_plot_dir, recursive = TRUE, showWarnings = FALSE)

# --- Function to calculate Standard Error ---
se <- function(x) {
  sd(x, na.rm = TRUE) / sqrt(length(na.omit(x)))
}


# --- 2. DATA LOADING AND PROCESSING ---
if (!file.exists(input_csv_path)) {
  stop(paste("Input file not found:", input_csv_path))
}

cat("Loading data from:", input_csv_path, "\n")
full_data <- read_csv(input_csv_path, col_types = cols(
  checkpoint_step = col_integer(),
  .default = col_character()
)) %>%
  mutate(across(c(null_hotspot_surprisal, overt_hotspot_surprisal, difference_score), as.numeric))


# --- Summarize data for plotting ---
cat("Summarizing data by checkpoint and source file...\n")
summary_data <- full_data %>%
  group_by(checkpoint_step, source_file) %>%
  summarise(
    mean_null = mean(null_hotspot_surprisal, na.rm = TRUE),
    se_null = se(null_hotspot_surprisal),
    mean_overt = mean(overt_hotspot_surprisal, na.rm = TRUE),
    se_overt = se(overt_hotspot_surprisal),
    mean_diff = mean(difference_score, na.rm = TRUE),
    se_diff = se(difference_score),
    .groups = 'drop'
  ) %>%
  pivot_longer(
    cols = c(mean_null, mean_overt),
    names_to = "condition",
    names_prefix = "mean_",
    values_to = "mean_surprisal"
  ) %>%
  mutate(
    se_surprisal = if_else(condition == "null", se_null, se_overt),
    condition = factor(condition, levels = c("null", "overt"))
  )

# --- 3. GENERATE LINE PLOTS (Surprisal over Time) ---
source_files <- sort(unique(summary_data$source_file))

cat("Generating surprisal-over-time plots for each source file...\n")
for (s_file in source_files) {

  plot_data <- summary_data %>% filter(source_file == s_file)

  p <- ggplot(plot_data, aes(x = checkpoint_step, y = mean_surprisal, color = condition, group = condition)) +
    geom_ribbon(aes(ymin = mean_surprisal - se_surprisal, ymax = mean_surprisal + se_surprisal, fill = condition), alpha = 0.2, linetype = 0) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    scale_color_manual(values = c("null" = "#0072B2", "overt" = "#D55E00")) +
    scale_fill_manual(values = c("null" = "#0072B2", "overt" = "#D55E00")) +
    labs(
      title = paste("Mean Hotspot Surprisal over Training"),
      subtitle = paste("Phenomenon:", s_file),
      x = "Training Step (Checkpoint)",
      y = "Mean Surprisal (Lower is Better)",
      color = "Pronoun Type",
      fill = "Pronoun Type"
    ) +
    theme_minimal(base_size = 14) +
    theme(legend.position = "bottom")

  output_filename <- file.path(line_plot_dir, paste0(str_remove(s_file, ".csv"), ".png"))
  ggsave(output_filename, plot = p, width = 10, height = 7, bg = "white")
}
cat("Saved", length(source_files), "line plots to:", line_plot_dir, "\n")


# --- 4. GENERATE FOREST PLOT (Final Checkpoint) ---
cat("Generating forest plot for the final checkpoint...\n")

final_checkpoint_step <- max(full_data$checkpoint_step, na.rm = TRUE)

forest_data <- full_data %>%
  filter(checkpoint_step == final_checkpoint_step) %>%
  group_by(source_file) %>%
  summarise(
    mean_diff = mean(difference_score, na.rm = TRUE),
    se_diff = se(difference_score),
    .groups = 'drop'
  ) %>%
  # CORRECTED: Set factor levels to be sorted alphanumerically.
  # The y-axis in ggplot is plotted from the bottom up, so we sort descending
  # to get an ascending alphanumeric plot (e.g., 1a, 1b, 2a...).
  mutate(source_file = factor(source_file, levels = sort(unique(source_file), decreasing = TRUE)))

forest_plot <- ggplot(forest_data, aes(x = mean_diff, y = source_file)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = mean_diff - se_diff, xmax = mean_diff + se_diff), height = 0.2) +
  geom_point(size = 4, color = "navy") +
  labs(
    title = "Pronoun Surprisal Difference at Final Checkpoint",
    subtitle = paste0("Model: ", model_tag, " | Checkpoint: ", final_checkpoint_step, "\n(Null Surprisal - Overt Surprisal)"),
    x = "Mean Difference in Surprisal (Negative favors overt pronoun)",
    y = "Evaluation Case"
  ) +
  theme_minimal(base_size = 14)

forest_plot_filename <- file.path(forest_plot_dir, "final_checkpoint_difference_forest_plot.png")
ggsave(forest_plot_filename, plot = forest_plot, width = 11, height = 8, bg = "white")
cat("Saved forest plot to:", forest_plot_filename, "\n")

cat("Analysis complete.\n")