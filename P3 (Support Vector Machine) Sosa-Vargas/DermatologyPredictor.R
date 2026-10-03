# ============================================================
# SVM desde cero con Kernel Trick - Dataset Dermatology
# ============================================================

# ── 1. Cargar datos ──────────────────────────────────────────
data_raw <- dermatology.data

col_names <- c(
  "erythema", "scaling", "definite_borders", "itching",
  "koebner_phenomenon", "polygonal_papules", "follicular_papules",
  "oral_mucosal_involvement", "knee_elbow_involvement", "scalp_involvement",
  "family_history", "melanin_incontinence", "eosinophils_infiltrate",
  "PNL_infiltrate", "fibrosis_papillary_dermis", "exocytosis",
  "acanthosis", "hyperkeratosis", "parakeratosis", "clubbing_rete_ridges",
  "elongation_rete_ridges", "thinning_suprapapillary", "spongiform_pustule",
  "munro_microabcess", "focal_hypergranulosis", "disappearance_granular",
  "vacuolisation_basal", "spongiosis", "sawtooth_retes",
  "follicular_horn_plug", "perifollicular_parakeratosis",
  "inflammatory_mononuclear", "band_like_infiltrate", "age", "class"
)
colnames(data_raw) <- col_names

# ── 2. Preprocesamiento ──────────────────────────────────────

# Forzar todas las columnas a numérico (los '?' quedan como NA)
for (col in col_names) {
  data_raw[[col]] <- as.numeric(as.character(data_raw[[col]]))
}

cat("NAs por columna:\n")
print(colSums(is.na(data_raw)))

# Imputar edad faltante con la mediana
data_raw$age[is.na(data_raw$age)] <- median(data_raw$age, na.rm = TRUE)

# Separar features y etiquetas
X_all <- as.matrix(data_raw[, 1:34])
y_all <- data_raw$class

cat("\nDimensiones:", dim(X_all), "\n")
cat("Clases únicas:", sort(unique(y_all)), "\n")
cat("Distribución de clases:\n")
print(table(y_all))

# Normalización min-max
min_vals   <- apply(X_all, 2, min,  na.rm = TRUE)
max_vals   <- apply(X_all, 2, max,  na.rm = TRUE)
range_vals <- max_vals - min_vals
range_vals[range_vals == 0] <- 1

X_scaled <- sweep(sweep(X_all, 2, min_vals, "-"), 2, range_vals, "/")

# ── 3. División 80 / 20 ──────────────────────────────────────
set.seed(42)
n         <- nrow(X_scaled)
idx_train <- sample(seq_len(n), size = floor(0.8 * n))
idx_test  <- setdiff(seq_len(n), idx_train)

X_train <- X_scaled[idx_train, ]
y_train <- y_all[idx_train]
X_test  <- X_scaled[idx_test, ]
y_test  <- y_all[idx_test]

cat("\nEntrenamiento:", nrow(X_train), "muestras\n")
cat("Prueba:       ", nrow(X_test),  "muestras\n\n")

# ============================================================
# FUNCIONES KERNEL
# ============================================================

kernel_linear <- function(x, z) {
  as.numeric(x %*% z)
}

kernel_poly <- function(x, z, gamma = 1, r = 0, d = 3) {
  (gamma * as.numeric(x %*% z) + r)^d
}

kernel_rbf <- function(x, z, gamma = 0.5) {
  diff <- x - z
  exp(-gamma * sum(diff^2))
}

kernel_sigmoid <- function(x, z, gamma = 0.01, r = 0) {
  tanh(gamma * as.numeric(x %*% z) + r)
}

# ── Matriz de Gram ────────────────────────────────────────────
compute_gram <- function(X, kernel_fn, ...) {
  n <- nrow(X)
  K <- matrix(0, n, n)
  for (i in seq_len(n)) {
    for (j in i:n) {
      val      <- kernel_fn(X[i, ], X[j, ], ...)
      K[i, j]  <- val
      K[j, i]  <- val
    }
  }
  K
}

# ============================================================
# SVM BINARIO
# ============================================================

svm_train_binary <- function(X, y, kernel_fn, C = 1,
                             max_iter = 300, lr = 0.005, ...) {
  n     <- nrow(X)
  K     <- compute_gram(X, kernel_fn, ...)
  Q     <- outer(y, y) * K
  alpha <- rep(0, n)
  
  for (iter in seq_len(max_iter)) {
    grad  <- rep(1, n) - Q %*% alpha
    alpha <- alpha + lr * grad
    alpha <- pmin(pmax(alpha, 0), C)
    # Proyección sobre Σ αᵢ yᵢ = 0
    alpha <- alpha - y * (sum(alpha * y) / sum(y * y))
    alpha <- pmin(pmax(alpha, 0), C)
  }
  
  sv_idx <- which(alpha > 1e-4)
  
  # Sesgo b
  b_vals <- numeric(length(sv_idx))
  for (k in seq_along(sv_idx)) {
    i   <- sv_idx[k]
    f_i <- 0
    for (j in sv_idx) {
      args <- c(list(X[j, ], X[i, ]), list(...))
      kval <- do.call(kernel_fn, args)
      f_i  <- f_i + alpha[j] * y[j] * kval
    }
    b_vals[k] <- y[i] - f_i
  }
  b <- mean(b_vals)
  
  list(
    alpha       = alpha,
    b           = b,
    sv_idx      = sv_idx,
    X_sv        = X[sv_idx, , drop = FALSE],
    y_sv        = y[sv_idx],
    alpha_sv    = alpha[sv_idx],
    kernel_fn   = kernel_fn,
    kernel_args = list(...)
  )
}

# ── Función de decisión ───────────────────────────────────────
# ŷ(x) = sgn( Σᵢ∈SV αᵢ yᵢ K(xᵢ,x) + b )
svm_predict_binary <- function(model, X_new) {
  scores <- apply(X_new, 1, function(xnew) {
    sv_scores <- numeric(length(model$alpha_sv))
    for (k in seq_along(model$alpha_sv)) {
      xsv  <- model$X_sv[k, ]
      args <- c(list(xsv, xnew), model$kernel_args)
      kval <- do.call(model$kernel_fn, args)
      sv_scores[k] <- model$alpha_sv[k] * model$y_sv[k] * kval
    }
    sum(sv_scores) + model$b
  })
  list(scores = scores, labels = sign(scores))
}

# ============================================================
# CLASIFICACIÓN MULTICLASE — One-vs-Rest
#
# Para c = 1…6:  yᵢ = +1 si clase c, -1 si no
# Asignar:  ŷ(x) = argmax_c f_c(x)
# ============================================================

svm_train_ovr <- function(X, y, kernel_fn, C = 1,
                          max_iter = 300, lr = 0.005, ...) {
  classes <- sort(unique(y))
  models  <- vector("list", length(classes))
  
  for (k in seq_along(classes)) {
    c_k   <- classes[k]
    y_bin <- ifelse(y == c_k, 1, -1)
    cat(sprintf("  Entrenando clasificador clase %d vs resto...\n", c_k))
    models[[k]] <- svm_train_binary(X, y_bin, kernel_fn, C,
                                    max_iter, lr, ...)
  }
  list(models = models, classes = classes)
}

svm_predict_ovr <- function(ovr_model, X_new) {
  # Matriz de puntuaciones: filas = obs, columnas = clases
  score_matrix <- matrix(0, nrow = nrow(X_new),
                         ncol = length(ovr_model$classes))
  for (k in seq_along(ovr_model$models)) {
    score_matrix[, k] <- svm_predict_binary(ovr_model$models[[k]],
                                            X_new)$scores
  }
  predicted_idx <- apply(score_matrix, 1, which.max)
  ovr_model$classes[predicted_idx]
}

# ── Métricas ──────────────────────────────────────────────────
accuracy <- function(y_true, y_pred) mean(y_true == y_pred)

confusion_matrix_manual <- function(y_true, y_pred, classes) {
  C   <- length(classes)
  mat <- matrix(0, C, C,
                dimnames = list(paste0("Real_", classes),
                                paste0("Pred_", classes)))
  for (i in seq_along(y_true)) {
    r  <- which(classes == y_true[i])
    cc <- which(classes == y_pred[i])
    mat[r, cc] <- mat[r, cc] + 1
  }
  mat
}

# ============================================================
# EXPERIMENTO: comparar los 4 kernels
# ============================================================

kernels <- list(
  lineal     = list(fn = kernel_linear),
  polinomico = list(fn = kernel_poly,    gamma = 1,    r = 0,  d = 3),
  rbf        = list(fn = kernel_rbf,     gamma = 0.5),
  sigmoidal  = list(fn = kernel_sigmoid, gamma = 0.01, r = 0)
)

results <- list()

for (kernel_name in names(kernels)) {
  cat(sprintf("\n=== Kernel: %s ===\n", kernel_name))
  kdef <- kernels[[kernel_name]]
  fn   <- kdef$fn
  args <- kdef[names(kdef) != "fn"]
  
  ovr <- do.call(svm_train_ovr,
                 c(list(X        = X_train,
                        y        = y_train,
                        kernel_fn = fn,
                        C        = 1,
                        max_iter = 300,
                        lr       = 0.005),
                   args))
  
  preds <- svm_predict_ovr(ovr, X_test)
  acc   <- accuracy(y_test, preds)
  cat(sprintf("  Exactitud en test: %.2f%%\n", acc * 100))
  
  results[[kernel_name]] <- list(model = ovr, preds = preds, acc = acc)
}

# ============================================================
# RESULTADOS FINALES
# ============================================================

cat("\n\n========================================\n")
cat("        RESUMEN DE RESULTADOS\n")
cat("========================================\n")
cat(sprintf("%-15s %s\n", "Kernel", "Exactitud (%)"))
cat("----------------------------------------\n")
for (nm in names(results)) {
  cat(sprintf("%-15s %.2f\n", nm, results[[nm]]$acc * 100))
}

best_kernel <- names(which.max(sapply(results, `[[`, "acc")))
cat(sprintf("\nMejor kernel: %s (%.2f%%)\n",
            best_kernel,
            results[[best_kernel]]$acc * 100))

# Matriz de confusión del mejor kernel
cat(sprintf("\nMatriz de confusion — Kernel %s:\n", best_kernel))
classes <- sort(unique(y_all))
cm <- confusion_matrix_manual(y_test,
                              results[[best_kernel]]$preds,
                              classes)
print(cm)

# Exactitud por clase
cat("\nExactitud por clase:\n")
etiquetas <- c("1 psoriasis", "2 seboreic dermatitis",
               "3 lichen planus", "4 pityriasis rosea",
               "5 cronic dermatitis", "6 pityriasis rubra pilaris")
for (i in seq_along(classes)) {
  cl   <- classes[i]
  mask <- y_test == cl
  if (any(mask)) {
    acc_cl <- mean(results[[best_kernel]]$preds[mask] == cl)
    cat(sprintf("  %s: %.2f%%\n", etiquetas[i], acc_cl * 100))
  }
}