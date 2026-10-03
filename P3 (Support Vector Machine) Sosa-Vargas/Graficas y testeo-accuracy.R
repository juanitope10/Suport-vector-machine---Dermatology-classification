# ============================================================
# VISUALIZACIONES - SVM Dermatology
# ============================================================

# ── Paleta de colores por clase ──────────────────────────────
class_colors <- c("#E63946", "#2A9D8F", "#E9C46A",
                  "#F4A261", "#457B9D", "#6A0572")
class_labels <- c("Psoriasis", "Seboreic Dermatitis",
                  "Lichen Planus", "Pityriasis Rosea",
                  "Cronic Dermatitis", "Pityriasis Rubra Pilaris")

# ============================================================
# GRÁFICA 1: Comparación de Accuracies por Kernel
# ============================================================
acc_vals <- sapply(results, `[[`, "acc") * 100
kernel_names <- c("Lineal", "Polinómico\n(d=3)", "RBF\n(γ=0.5)", "Sigmoidal")

par(mar = c(5, 5, 4, 2))
bp <- barplot(
  acc_vals,
  names.arg = kernel_names,
  col       = c("#457B9D", "#E9C46A", "#2A9D8F", "#F4A261"),
  border    = NA,
  ylim      = c(0, 110),
  main      = "Exactitud por Kernel — Test Set (80/20)",
  ylab      = "Exactitud (%)",
  xlab      = "Función Kernel",
  cex.main  = 1.3,
  cex.lab   = 1.1,
  las       = 1
)
# Etiquetas sobre las barras
text(bp, acc_vals + 2,
     labels = sprintf("%.2f%%", acc_vals),
     cex = 1.0, font = 2)
# Línea de referencia
abline(h = 100, lty = 2, col = "gray50")
abline(h = 50,  lty = 3, col = "gray70")

# ============================================================
# GRÁFICA 2: Heatmap — Matriz de Confusión (RBF)
# ============================================================
cm_rbf <- confusion_matrix_manual(y_test,
                                  results[["rbf"]]$preds,
                                  classes)

# Normalizar por fila (% sobre la clase real)
cm_norm <- cm_rbf / rowSums(cm_rbf) * 100

par(mar = c(6, 8, 4, 2))
image(
  t(cm_norm[nrow(cm_norm):1, ]),
  col    = colorRampPalette(c("white", "#2A9D8F"))(100),
  axes   = FALSE,
  main   = "Matriz de Confusión Normalizada — Kernel RBF (%)",
  cex.main = 1.2
)

# Ejes
axis(1, at = seq(0, 1, length.out = 6),
     labels = class_labels, las = 2, cex.axis = 0.75)
axis(2, at = seq(0, 1, length.out = 6),
     labels = rev(class_labels), las = 1, cex.axis = 0.75)

# Valores en cada celda
for (i in 1:6) {
  for (j in 1:6) {
    val <- cm_norm[i, j]
    text_col <- if (val > 60) "white" else "gray20"
    text(
      x   = (j - 1) / 5,
      y   = (6 - i) / 5,
      labels = sprintf("%.0f%%", val),
      cex    = 1.0,
      col    = text_col,
      font   = 2
    )
  }
}
box()

# ============================================================
# GRÁFICA 3: PCA — Datos en 2D coloreados por clase
# ============================================================

# PCA manual (sin prcomp si se quiere, pero es base R — sin librerías extra)
pca_result <- prcomp(X_scaled, center = TRUE, scale. = FALSE)
PC <- pca_result$x[, 1:2]

var_exp <- round(pca_result$sdev[1:2]^2 / sum(pca_result$sdev^2) * 100, 1)

par(mar = c(5, 5, 4, 8), xpd = TRUE)
plot(
  PC[, 1], PC[, 2],
  col  = class_colors[y_all],
  pch  = 19,
  cex  = 0.8,
  main = "PCA — Espacio original (2 componentes principales)",
  xlab = sprintf("PC1 (%.1f%% varianza)", var_exp[1]),
  ylab = sprintf("PC2 (%.1f%% varianza)", var_exp[2]),
  cex.main = 1.2,
  cex.lab  = 1.1
)
legend("topright", inset = c(-0.35, 0),
       legend = class_labels,
       col    = class_colors,
       pch    = 19, cex = 0.7,
       title  = "Clase", bty = "n")

# ============================================================
# GRÁFICA 4: Frontera de decisión RBF en espacio PCA (2D)
# ============================================================

# Proyectar train y test al espacio PCA
PC_train <- pca_result$x[idx_train, 1:2]
PC_test  <- pca_result$x[idx_test,  1:2]

# Grid sobre el espacio PCA
x_range <- range(PC[, 1])
y_range <- range(PC[, 2])
grid_n  <- 80   # resolución (aumentar = más lento)

gx <- seq(x_range[1], x_range[2], length.out = grid_n)
gy <- seq(y_range[1], y_range[2], length.out = grid_n)
grid_pts <- as.matrix(expand.grid(gx, gy))

# Entrenar SVM RBF sobre las 2 PCs
cat("\nEntrenando SVM RBF sobre PCA 2D para visualización...\n")
ovr_pca <- svm_train_ovr(
  X        = PC_train,
  y        = y_train,
  kernel_fn = kernel_rbf,
  C        = 1,
  max_iter = 300,
  lr       = 0.005,
  gamma    = 0.5
)

# Predecir en el grid
grid_preds <- svm_predict_ovr(ovr_pca, grid_pts)

# Paleta translúcida para el fondo
bg_colors <- adjustcolor(class_colors, alpha.f = 0.25)

par(mar = c(5, 5, 4, 8), xpd = TRUE)
plot(
  grid_pts[, 1], grid_pts[, 2],
  col  = bg_colors[grid_preds],
  pch  = 15,
  cex  = 0.55,
  main = "Frontera de Decisión — Kernel RBF (espacio PCA 2D)",
  xlab = sprintf("PC1 (%.1f%% varianza)", var_exp[1]),
  ylab = sprintf("PC2 (%.1f%% varianza)", var_exp[2]),
  cex.main = 1.2,
  cex.lab  = 1.1
)

# Puntos de entrenamiento
points(PC_train[, 1], PC_train[, 2],
       col = class_colors[y_train], pch = 19, cex = 0.7)

# Puntos de test con borde negro para distinguirlos
points(PC_test[, 1], PC_test[, 2],
       col = class_colors[y_test], pch = 21,
       bg  = class_colors[y_test], cex = 1.1, lwd = 1.5)

legend("topright", inset = c(-0.35, 0),
       legend = c(class_labels, "", "Train", "Test"),
       col    = c(class_colors, NA, "gray30", "gray30"),
       pch    = c(rep(19, 6), NA, 19, 21),
       cex    = 0.7, title = "Clase", bty = "n")

# ============================================================
# GRÁFICA 5: Exactitud por clase — todos los kernels
# ============================================================
kernel_display <- c("Lineal", "Polinómico", "RBF", "Sigmoidal")
acc_by_class <- matrix(NA, nrow = 6, ncol = 4,
                       dimnames = list(class_labels, kernel_display))

for (k in seq_along(names(results))) {
  nm    <- names(results)[k]
  preds <- results[[nm]]$preds
  for (i in seq_along(classes)) {
    cl   <- classes[i]
    mask <- y_test == cl
    if (any(mask)) {
      acc_by_class[i, k] <- mean(preds[mask] == cl) * 100
    }
  }
}

par(mar = c(8, 5, 4, 2))
barplot(
  t(acc_by_class),
  beside     = TRUE,
  col        = c("#457B9D", "#E9C46A", "#2A9D8F", "#F4A261"),
  border     = NA,
  ylim       = c(0, 120),
  main       = "Exactitud por Clase y Kernel",
  ylab       = "Exactitud (%)",
  names.arg  = class_labels,
  las        = 2,
  cex.names  = 0.72,
  cex.main   = 1.2,
  legend.text = kernel_display,
  args.legend = list(x = "topright", bty = "n",
                     cex = 0.85, fill = c("#457B9D","#E9C46A","#2A9D8F","#F4A261"),
                     border = NA)
)
abline(h = 100, lty = 2, col = "gray50")


# ============================================================
# ANEXO: Validación estadística de los modelos entrenados
# Sin reentrenar — solo evaluación sobre múltiples submuestras
# ============================================================

# ── Parámetros de la validación ──────────────────────────────
N_ITER   <- 100    # número de submuestras de test
TEST_PROP <- 0.20  # proporción del total usada en cada submuestra

set.seed(123)

# Clases y etiquetas
classes   <- sort(unique(y_all))
etiquetas <- c("Psoriasis", "Seb. Dermatitis", "Lichen Planus",
               "Pityriasis Rosea", "Cronic Dermatitis", "Pit. Rubra Pil.")

# ============================================================
# FUNCIÓN: evaluar un modelo OvR sobre una submuestra aleatoria
# de los índices de TEST originales
# ============================================================
evaluate_subsample <- function(ovr_model, X_full, y_full,
                               idx_pool, prop, classes) {
  n_sub  <- floor(length(idx_pool) * prop / TEST_PROP)
  n_sub  <- min(n_sub, length(idx_pool))
  idx_sub <- sample(idx_pool, size = n_sub, replace = FALSE)
  
  X_sub <- X_full[idx_sub, , drop = FALSE]
  y_sub <- y_full[idx_sub]
  
  preds <- svm_predict_ovr(ovr_model, X_sub)
  acc   <- mean(preds == y_sub)
  
  # Exactitud por clase
  acc_cls <- sapply(classes, function(cl) {
    mask <- y_sub == cl
    if (sum(mask) == 0) return(NA)
    mean(preds[mask] == cl)
  })
  
  list(acc = acc, acc_cls = acc_cls, n = n_sub)
}

# ============================================================
# BUCLE PRINCIPAL: N_ITER submuestras por kernel
# ============================================================
cat("Ejecutando", N_ITER, "submuestras de evaluación por kernel...\n\n")

val_results <- list()

for (kernel_name in names(results)) {
  cat(sprintf("Evaluando kernel: %s\n", kernel_name))
  
  ovr_model <- results[[kernel_name]]$model
  
  acc_vec     <- numeric(N_ITER)
  acc_cls_mat <- matrix(NA, nrow = N_ITER, ncol = length(classes))
  
  for (i in seq_len(N_ITER)) {
    ev <- evaluate_subsample(
      ovr_model  = ovr_model,
      X_full     = X_scaled,
      y_full     = y_all,
      idx_pool   = idx_test,
      prop       = TEST_PROP,
      classes    = classes
    )
    acc_vec[i]      <- ev$acc
    acc_cls_mat[i,] <- ev$acc_cls
  }
  
  val_results[[kernel_name]] <- list(
    acc_vec     = acc_vec,
    acc_cls_mat = acc_cls_mat
  )
  
  cat(sprintf("  Media:  %.2f%%  |  SD: %.2f%%  |  Min: %.2f%%  |  Max: %.2f%%\n\n",
              mean(acc_vec) * 100,
              sd(acc_vec)   * 100,
              min(acc_vec)  * 100,
              max(acc_vec)  * 100))
}

# ============================================================
# TABLA RESUMEN
# ============================================================
cat("============================================================\n")
cat("  RESUMEN ESTADÍSTICO —", N_ITER, "submuestras de evaluación\n")
cat("============================================================\n")
cat(sprintf("%-15s %8s %8s %8s %8s %8s\n",
            "Kernel", "Media", "SD", "Min", "P25", "Max"))
cat(paste(rep("-", 60), collapse = ""), "\n")

for (nm in names(val_results)) {
  v <- val_results[[nm]]$acc_vec * 100
  cat(sprintf("%-15s %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%%\n",
              nm,
              mean(v), sd(v),
              min(v),
              quantile(v, 0.25),
              max(v)))
}

# ============================================================
# GRÁFICA 1: Boxplot de accuracies por kernel
# ============================================================
acc_list <- lapply(val_results, function(x) x$acc_vec * 100)

kernel_display <- c("Lineal", "Polinómico", "RBF", "Sigmoidal")
colores <- c("#457B9D", "#E9C46A", "#2A9D8F", "#F4A261")

par(mar = c(5, 5, 4, 2))
bp <- boxplot(
  acc_list,
  names   = kernel_display,
  col     = colores,
  border  = "gray30",
  ylim    = c(0, 105),
  main    = sprintf("Distribución de Exactitud — %d submuestras de test", N_ITER),
  ylab    = "Exactitud (%)",
  xlab    = "Kernel",
  cex.main = 1.2,
  cex.lab  = 1.1,
  las      = 1,
  outline  = TRUE,
  whisklty = 1
)

# Media como punto rojo
means <- sapply(acc_list, mean)
points(seq_along(acc_list), means, pch = 18, col = "red", cex = 1.5)

# Etiqueta de media
text(seq_along(acc_list), means + 3,
     labels = sprintf("%.1f%%", means),
     cex = 0.85, col = "red", font = 2)

abline(h = 100, lty = 2, col = "gray50")
legend("bottomright",
       legend = c("Mediana", "Media"),
       pch    = c(NA, 18),
       lty    = c(1, NA),
       col    = c("black", "red"),
       bty    = "n", cex = 0.9)

# ============================================================
# GRÁFICA 2: Histogramas de accuracy — RBF vs Lineal
# ============================================================
par(mfrow = c(1, 2), mar = c(5, 4, 4, 1))

# RBF
hist(val_results$rbf$acc_vec * 100,
     breaks  = 15,
     col     = "#2A9D8F",
     border  = "white",
     main    = "Kernel RBF",
     xlab    = "Exactitud (%)",
     ylab    = "Frecuencia",
     xlim    = c(80, 100),
     cex.main = 1.2)
abline(v = mean(val_results$rbf$acc_vec) * 100,
       col = "white", lwd = 2, lty = 2)
text(mean(val_results$rbf$acc_vec) * 100,
     par("usr")[4] * 0.9,
     labels = sprintf("Media\n%.1f%%", mean(val_results$rbf$acc_vec) * 100),
     col = "white", cex = 0.85, font = 2)

# Lineal
hist(val_results$lineal$acc_vec * 100,
     breaks  = 15,
     col     = "#457B9D",
     border  = "white",
     main    = "Kernel Lineal",
     xlab    = "Exactitud (%)",
     ylab    = "Frecuencia",
     xlim    = c(70, 100),
     cex.main = 1.2)
abline(v = mean(val_results$lineal$acc_vec) * 100,
       col = "white", lwd = 2, lty = 2)
text(mean(val_results$lineal$acc_vec) * 100,
     par("usr")[4] * 0.9,
     labels = sprintf("Media\n%.1f%%", mean(val_results$lineal$acc_vec) * 100),
     col = "white", cex = 0.85, font = 2)

par(mfrow = c(1, 1))

# ============================================================
# GRÁFICA 3: Exactitud por clase — RBF (todas las iteraciones)
# ============================================================
acc_cls_rbf <- val_results$rbf$acc_cls_mat * 100

par(mar = c(8, 5, 4, 2))
boxplot(
  acc_cls_rbf,
  names    = etiquetas,
  col      = "#2A9D8F",
  border   = "gray30",
  ylim     = c(50, 105),
  main     = sprintf("Exactitud por Clase — Kernel RBF (%d submuestras)", N_ITER),
  ylab     = "Exactitud (%)",
  xlab     = "",
  las      = 2,
  cex.axis = 0.8,
  cex.main = 1.2,
  outline  = TRUE
)
abline(h = 100, lty = 2, col = "gray50")
abline(h  = 90,  lty = 3, col = "gray70")

# Medias por clase
means_cls <- colMeans(acc_cls_rbf, na.rm = TRUE)
points(seq_along(etiquetas), means_cls,
       pch = 18, col = "red", cex = 1.4)

# ============================================================
# GRÁFICA 4: Evolución acumulada de la media (convergencia)
# ============================================================
par(mar = c(5, 5, 4, 2))
plot(NULL, xlim = c(1, N_ITER), ylim = c(60, 100),
     main = "Convergencia de la Media Acumulada de Exactitud",
     xlab = "Número de submuestras",
     ylab = "Media acumulada (%)",
     cex.main = 1.2, cex.lab = 1.1, las = 1)

kernel_cols <- c(lineal     = "#457B9D",
                 polinomico = "#E9C46A",
                 rbf        = "#2A9D8F",
                 sigmoidal  = "#F4A261")

for (nm in names(val_results)) {
  v        <- val_results[[nm]]$acc_vec * 100
  cum_mean <- cumsum(v) / seq_along(v)
  lines(seq_along(v), cum_mean,
        col = kernel_cols[nm], lwd = 2)
}

legend("bottomright",
       legend = kernel_display,
       col    = unname(kernel_cols),
       lwd    = 2, bty = "n", cex = 0.9)

abline(h = 98.65, lty = 2, col = "#2A9D8F", lwd = 1)
text(N_ITER * 0.6, 99.5, "RBF test original (98.65%)",
     col = "#2A9D8F", cex = 0.8)

# ============================================================
# TEST DE WILCOXON: RBF vs cada otro kernel
# ============================================================
cat("\n============================================================\n")
cat("  TEST DE WILCOXON — RBF vs otros kernels\n")
cat("  H0: las distribuciones de exactitud son iguales\n")
cat("============================================================\n")

rbf_acc <- val_results$rbf$acc_vec

for (nm in c("lineal", "polinomico", "sigmoidal")) {
  other_acc <- val_results[[nm]]$acc_vec
  wt <- wilcox.test(rbf_acc, other_acc,
                    alternative = "greater",
                    paired      = TRUE,
                    exact       = FALSE)
  cat(sprintf("\nRBF vs %-12s | W = %6.1f | p-valor = %.2e | %s\n",
              nm,
              wt$statistic,
              wt$p.value,
              ifelse(wt$p.value < 0.05,
                     "RBF significativamente mejor",
                     "Sin diferencia significativa")))
}