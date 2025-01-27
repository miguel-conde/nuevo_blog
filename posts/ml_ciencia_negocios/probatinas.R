library(tidyverse)
library(tidymodels)

masa_sol <- 1.989e30 # kg
radio_sol <- 6.957e8 # m
masa_tierra <- 5.972e24 # kg
radio_tierra <- 6.371e6 # m
G <- 6.674e-11 # m^3 kg^-1 s^-2


stars_df <- read_csv("posts/ml_ciencia_negocios/data/estrellas_mas_luminosas.csv") %>% 
  janitor::clean_names() %>% 
  mutate(
    masa_kg = masa_m * masa_sol,
    radio_m = radio_r * radio_sol
  ) %>% 
  mutate(
    g_superficie = G * masa_kg / radio_m^2,
    g_sup_medida = g_superficie * (1 + rnorm(n(), 0, 0.1)),
  )


sist_solar_df <- read_csv("posts/ml_ciencia_negocios/data/cuerpos_mas_masivos_sistema_solar.csv") %>% 
  janitor::clean_names()%>% 
  mutate(
    masa_kg = masa_m * masa_tierra,
    radio_m = radio_r * radio_tierra
  ) %>% 
  mutate(
    g_superficie = G * masa_kg / radio_m^2,
    g_sup_medida = g_superficie * (1 + rnorm(n(), 0, 0.1)),
  )

#### Modelo Random Forest con tidymodels usando validación cruzada
# Definir partición para validación cruzada
set.seed(123)
cv_folds <- vfold_cv(stars_df, v = 10)

# Definir el modelo Random Forest
rf_model <- rand_forest(
  mode = "regression",
  mtry = 2,
  trees = 500
) %>%
  set_engine("ranger")

# Especificación del workflow
rf_workflow <- workflow() %>%
  add_formula(log(g_sup_medida) ~ log(masa_kg) + log(radio_m)) %>%
  add_model(rf_model)

# Realizar validación cruzada
ctrl <- control_resamples(save_pred = TRUE, save_workflow = TRUE)
rf_results <- fit_resamples(
  rf_workflow,
  resamples = cv_folds,
  metrics = metric_set(rmse, rsq),
  control = ctrl
)

# Mostrar los resultados
rf_results %>% collect_metrics()

# Obtener el mejor modelo de los resultados de validación cruzada
best_rf_model <- rf_results %>% 
  select_best(metric = "rmse")

# Ajustar el modelo con los mejores hiperparámetros en todos los datos
final_rf_workflow <- finalize_workflow(rf_workflow, best_rf_model)

final_rf_fit <- fit(final_rf_workflow, stars_df)

# Error de training
final_rf_fit %>% 
  predict(stars_df) %>% 
  bind_cols(stars_df) %>% 
  mutate(
    error = g_sup_medida - exp(.pred)
  ) %>% 
  summarise(
    rmse = sqrt(mean(error^2)),
    mae = mean(abs(error))
  )

#### Modelo lineal

g_lm <- lm(log(g_sup_medida) ~ log(masa_kg) + log(radio_m), data = stars_df)
summary(g_lm)

library(broom)

glance(g_lm)
tidy(g_lm, conf.int = TRUE)
augment(g_lm)

tidy(g_lm, conf.int = TRUE) %>% 
  filter(term == '(Intercept)') %>%
  select(estimate, conf.low, conf.high) %>% 
  mutate_all(exp)


### PREDICCIONES
# Predicciones con Random Forest
rf_predictions_stars <- predict(final_rf_fit, stars_df) %>%
  bind_cols(stars_df) %>% 
  mutate(
    g_sup_pred = exp(.pred)
  )

rf_predictions_stars %>% 
  ggplot(aes(g_sup_medida, g_sup_pred)) +
  geom_point() +
  geom_abline() +
  labs(
    title = "Predicciones de Gravedad en la Superficie - ESTRELLAS",
    x = "Gravedad Superficie Medida",
    y = "Gravedad Superficie Predicha"
  )

rf_predictions_stars %>% 
  mutate(
    error = g_sup_medida - g_sup_pred
  ) %>% 
  summarise(
    rmse = sqrt(mean(error^2)),
    mae = mean(abs(error))
  )

rf_predictions_sist_solar <- predict(final_rf_fit, sist_solar_df) %>%
  bind_cols(sist_solar_df) %>% 
  mutate(
    g_sup_pred = exp(.pred)
  )

rf_predictions_sist_solar %>% 
  ggplot(aes(g_sup_medida, g_sup_pred)) +
  geom_point() +
  geom_abline() +
  labs(
    title = "Predicciones de Gravedad en la Superficie - SISTEMA SOLAR",
    x = "Gravedad Superficie Medida",
    y = "Gravedad Superficie Predicha"
  )

# Calculamos el error de prediccion
rf_predictions_sist_solar %>% 
  mutate(
    error = g_sup_medida - g_sup_pred
  ) %>% 
  summarise(
    rmse = sqrt(mean(error^2)),
    mae = mean(abs(error))
  )

# Predicciones con modelo lineal
lm_predictions_estrellas <- augment(g_lm, newdata = stars_df) %>% 
  mutate(
    g_sup_pred = exp(.fitted)
  )

lm_predictions_estrellas %>%
  ggplot(aes(g_sup_medida, g_sup_pred)) +
  geom_point() +
  geom_abline() +
  labs(
    title = "Predicciones de Gravedad en la Superficie - ESTRELLAS",
    x = "Gravedad Superficie Medida",
    y = "Gravedad Superficie Predicha"
  )

# Calculamos el error de prediccion
lm_predictions_estrellas %>% 
  mutate(
    error = g_sup_medida - g_sup_pred
  ) %>% 
  summarise(
    rmse = sqrt(mean(error^2)),
    mae = mean(abs(error))
  )


lm_predictions_sist_solar <- augment(g_lm, newdata = sist_solar_df) %>% 
  mutate(
    g_sup_pred = exp(.fitted)
  )

lm_predictions_sist_solar %>%
  ggplot(aes(g_sup_medida, g_sup_pred)) +
  geom_point() +
  geom_abline() +
  labs(
    title = "Predicciones de Gravedad en la Superficie - SISTEMA SOLAR",
    x = "Gravedad Superficie Medida",
    y = "Gravedad Superficie Predicha"
  )

# Calculamos el error de prediccion
lm_predictions_sist_solar %>% 
  mutate(
    error = g_sup_medida - g_sup_pred
  ) %>% 
  summarise(
    rmse = sqrt(mean(error^2)),
    mae = mean(abs(error))
  )
