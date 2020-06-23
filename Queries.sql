SELECT Models.model_name, COUNT(operator_type) AS num_conv2d_layers
FROM Models JOIN Operators
ON Models.model_name = Operators.model_name
WHERE operator_type = "CONV_2D"
GROUP BY model_name
ORDER BY num_conv2d_layers DESC;


SELECT category, operator_type, COUNT(operator_id) as count
FROM Models JOIN Operators
ON Models.model_name = Operators.model_name
WHERE operator_type NOT IN ("Output_Placeholder", "Input_Placeholder")
GROUP BY category, operator_type
ORDER BY category, count DESC;


SELECT Models.model_name, activation_function, COUNT(operator_type) as count
FROM Models JOIN Operators
ON Models.model_name = Operators.model_name
GROUP BY model_name, activation_function
ORDER BY model_name, count DESC;


SELECT Models.model_name, operator_type, padding, COUNT(operator_id) as count
FROM Models JOIN Operators
ON Models.model_name = Operators.model_name
WHERE operator_type NOT IN ("Output_Placeholder", "Input_Placeholder") AND padding IS NOT NULL
GROUP BY model_name, operator_type, padding
ORDER BY model_name, count DESC;

SELECT operator_type, COUNT(operator_id) as count
FROM Operators
WHERE operator_type 
IN ("Relu", "Relu6", "Sigmoid", "Elu","Exp", "Selu", "Softplus", "Softsign", "Tanh")
GROUP BY operator_type
ORDER BY count DESC;