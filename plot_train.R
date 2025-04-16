#!/usr/bin/env Rscript

# Load required libraries
library(ggplot2)

# Parse command line argument for table name (without extension)
args <- commandArgs(trailingOnly = TRUE)
if(length(args) < 1) {
  stop("Usage: script.R <table_name>")
}
table_name <- args[1]

# Read the data
data <- read.delim(table_name)



# Create a group column for plotting
data$group <- ifelse(is.na(data$val_loss), "train", "validation")

# Create the plot excluding step == 0, with specific aesthetic mapping and limits
p <- ggplot(subset(data, step != 0), aes(x = step, y = train_loss, color = group)) +
  geom_line() +
  labs(color = "group")  # Change legend title

# Save the plot as a PDF (5 x 4 inches)
output_file <- paste0(table_name, ".pdf")
ggsave(filename = output_file, plot = p, width = 8, height = 4, units = "in")
