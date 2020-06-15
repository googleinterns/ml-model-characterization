SELECT
  Operators.model_name,
  tensor_label,
  tensor_shape,
  tensor_type
FROM
  Operators
JOIN
  Tensors
ON
  Operators.model_name = Tensors.model_name
  AND operator_id IN UNNEST(from_operator_ids)
WHERE
  operator_type = "Input_Placeholder"

SELECT
  Models.model_name,
  COUNT(operator_type) AS num_conv2d_layers
FROM
  Models
JOIN
  Operators
ON
  Models.model_name = Operators.model_name
WHERE
  operator_type = "CONV_2D"
GROUP BY
  model_name

SELECT
  category,
  operator_type,
  COUNT(operator_id)
FROM
  Models
JOIN
  Operators
ON
  Models.model_name = Operators.model_name
WHERE
  operator_type NOT IN ("Output_Placeholder",
    "Input_Placeholder")
GROUP BY
  category,
  operator_type
ORDER BY
  category

SELECT
  Models.model_name,
  activation_function,
  COUNT(operator_type)
FROM
  Models
JOIN
  Operators
ON
  Models.model_name = Operators.model_name
GROUP BY
  model_name,
  activation_function

SELECT
  Models.model_name,
  operator_type,
  padding,
  COUNT(operator_id)
FROM
  Models
JOIN
  Operators
ON
  Models.model_name = Operators.model_name
WHERE
  operator_type NOT IN ("Output_Placeholder",
    "Input_Placeholder")
  AND padding IS NOT NULL
GROUP BY
  model_name,
  operator_type,
  padding
ORDER BY
  model_name