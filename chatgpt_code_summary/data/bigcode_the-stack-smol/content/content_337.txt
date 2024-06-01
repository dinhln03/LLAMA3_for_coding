def gen_mutants():
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    import tensorflow as tf
    import pandas
    import numpy as np
    
    
    
    
    
    
    DATAFILE_TRAIN = 'mock_kaggle_edit_train.csv'
    
    DATAFILE_VALIDATE = 'mock_kaggle_edit_validate.csv'
    
    
    
    
    
    TRAINED_MODEL_PATH = 'savedModel'
    
    TIME_STEPS = 10
    NUMBER_OF_DAYS_TO_FORECAST = 1
    
    BATCH_SIZE = 100
    
    NUM_EPOCHS = 100
    
    LSTM_UNITS = 250
    
    TENSORBOARD_LOGDIR = 'tensorboard_log'
    
    
    
    
    
    
    data_train = pandas.read_csv(DATAFILE_TRAIN)
    data_validate = pandas.read_csv(DATAFILE_VALIDATE)
    
    
    
    
    
    
    data_train.head()
    
    
    
    
    
    
    
    numTrainingData = len(data_train)
    numValidationData = len(data_validate)
    
    trainingData_date = data_train['date'][0:numTrainingData]
    trainingData_sales = data_train['sales'][0:numTrainingData]
    trainindData_price = data_train['price'][0:numTrainingData]
    
    validationData_date = data_validate['date'][0:numValidationData]
    validationData_sales = data_validate['sales'][0:numValidationData]
    validationData_price = data_validate['price'][0:numValidationData]
    
    
    
    
    
    trainingData_sales.head()
    
    
    
    
    
    print(len(trainingData_sales))
    print(len(validationData_sales))
    
    
    
    
    
    
    
    
    trainingData_sales_min = min(trainingData_sales)
    trainingData_sales_max = max(trainingData_sales)
    trainingData_sales_range = trainingData_sales_max - trainingData_sales_min
    trainingData_sales_normalised = [(i - trainingData_sales_min) / trainingData_sales_range for i in trainingData_sales]
    
    validationData_sales_normalised = [(i - trainingData_sales_min) / trainingData_sales_range for i in validationData_sales]
    
    
    
    
    
    
    print('Min:', trainingData_sales_min)
    print('Range:', trainingData_sales_max - trainingData_sales_min)
    
    
    
    
    
    
    trainingDataSequence_sales = np.zeros(shape=(((len(trainingData_sales) - TIME_STEPS) - NUMBER_OF_DAYS_TO_FORECAST) + 1, TIME_STEPS, 1))
    targetDataSequence_sales = np.zeros(shape=(((len(trainingData_sales) - TIME_STEPS) - NUMBER_OF_DAYS_TO_FORECAST) + 1, NUMBER_OF_DAYS_TO_FORECAST))
    start = 0
    for i in range(TIME_STEPS, (len(trainingData_sales) - NUMBER_OF_DAYS_TO_FORECAST) + 1):
        trainingDataSequence_sales[start,:,0] = trainingData_sales_normalised[start:i]
        targetDataSequence_sales[start] = trainingData_sales_normalised[i:]
        start = start + 1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    [trainingDataSequence_sales[i,:,0] for i in range(3)]
    
    
    
    
    
    
    [targetDataSequence_sales[i] for i in range(3)]
    
    
    
    
    
    
    
    
    
    
    
    
    a = np.arange(len(targetDataSequence_sales))
    np.random.shuffle(a)
    trainingDataSequence_sales_shuffle = np.zeros(shape=(((len(trainingData_sales) - TIME_STEPS) - NUMBER_OF_DAYS_TO_FORECAST) + 1, TIME_STEPS, 1))
    targetDataSequence_sales_shuffle = np.zeros(shape=(((len(trainingData_sales) - TIME_STEPS) - NUMBER_OF_DAYS_TO_FORECAST) + 1, NUMBER_OF_DAYS_TO_FORECAST))
    
    loc = 0
    for i in a:
        trainingDataSequence_sales_shuffle[loc] = trainingDataSequence_sales[i]
        targetDataSequence_sales_shuffle[loc] = targetDataSequence_sales[i]
        loc += 1
    
    trainingDataSequence_sales = trainingDataSequence_sales_shuffle
    targetDataSequence_sales = targetDataSequence_sales_shuffle
    
    
    
    
    
    
    validationDataSequence_sales = np.zeros(shape=(((len(validationData_sales) - TIME_STEPS) - NUMBER_OF_DAYS_TO_FORECAST) + 1, TIME_STEPS, 1))
    validationDataSequence_sales_target = np.zeros(shape=(((len(validationData_sales) - TIME_STEPS) - NUMBER_OF_DAYS_TO_FORECAST) + 1, NUMBER_OF_DAYS_TO_FORECAST))
    
    start = 0
    for i in range(TIME_STEPS, (len(validationData_sales) - NUMBER_OF_DAYS_TO_FORECAST) + 1):
        validationDataSequence_sales[start,:,0] = validationData_sales_normalised[start:i]
        validationDataSequence_sales_target[start] = validationData_sales_normalised[i:i + NUMBER_OF_DAYS_TO_FORECAST]
        start += 1
    
    
    
    
    
    
    tf.reset_default_graph()
    
    inputSequencePlaceholder = tf.placeholder(dtype=tf.float32, shape=(None, TIME_STEPS, 1), name='inputSequencePlaceholder')
    targetPlaceholder = tf.placeholder(dtype=tf.float32, shape=(None, NUMBER_OF_DAYS_TO_FORECAST), name='targetPlaceholder')
    
    
    cell = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_UNITS, name='LSTM_cell')
    
    
    (output, state) = tf.nn.dynamic_rnn(cell=cell, inputs=inputSequencePlaceholder, dtype=tf.float32)
    
    
    lastCellOutput = output[:,-1,:]
    
    
    
    
    
    print('output:', output)
    print('state:', state)
    print('lastCellOutput:', lastCellOutput)
    
    
    
    
    
    
    
    
    
    
    
    
    
    weights = tf.Variable(initial_value=tf.truncated_normal(shape=(LSTM_UNITS, NUMBER_OF_DAYS_TO_FORECAST)))
    bias = tf.Variable(initial_value=tf.ones(shape=NUMBER_OF_DAYS_TO_FORECAST))
    
    forecast = tf.add(x=tf.matmul(a=lastCellOutput, b=weights), y=bias, name='forecast_normalised_scale')
    
    
    
    
    forecast_originalScale = tf.add(x=forecast * trainingData_sales_range, y=trainingData_sales_min, name='forecast_original_scale')
    
    
    
    
    
    print(forecast)
    print(forecast_originalScale)
    
    
    
    
    
    
    
    loss = tf.reduce_mean(tf.squared_difference(x=forecast, y=targetPlaceholder), name='loss_comp')
    
    tf.summary.scalar(tensor=loss, name='loss')
    
    
    
    
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    minimize_step = optimizer.minimize(loss)
    
    
    
    
    
    
    
    
    
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        
        tensorboard_writer = tf.summary.FileWriter(TENSORBOARD_LOGDIR, sess.graph)
        
        
        all_summary_ops = tf.summary.merge_all()
        
        
        numSteps = 0
        for e in range(NUM_EPOCHS):
            print('starting training for epoch:', e + 1)
            
            startLocation = 0
            iteration = 0
            for iteration in range(int(len(targetDataSequence_sales) / BATCH_SIZE)):
                print('epoch:', e + 1, ' iteration:', iteration + 1)
                trainingBatchInput = trainingDataSequence_sales[startLocation:startLocation + BATCH_SIZE,:,:]
                trainingBatchTarget = targetDataSequence_sales[startLocation:startLocation + BATCH_SIZE]
                
                (_, lsBatch, forecastBatch, forecastBatch_originalScale, summary_values) = sess.run([minimize_step, loss, forecast, forecast_originalScale, all_summary_ops], feed_dict={inputSequencePlaceholder: trainingBatchInput, \
                    targetPlaceholder: trainingBatchTarget})
                
                tensorboard_writer.add_summary(summary_values, numSteps)
                numSteps += 1
                
                if (iteration + 1) % 1 == 0:
                    print('got a loss of:', lsBatch)
                    print('the forecast of first 5 normalised are:', forecastBatch[0:5])
                    print('while the actuals were normalised     :', trainingBatchTarget[0:5])
                    print('the forecast of first 5 orignal scale are:', forecastBatch_originalScale[0:5])
                    print('while the actuals were original scale     :', (trainingBatchTarget[0:5] * trainingData_sales_range) + trainingData_sales_min)
                
                startLocation += BATCH_SIZE
            
            
            if len(targetDataSequence_sales) > startLocation:
                print('epoch:', e + 1, ' iteration:', iteration + 1)
                trainingBatchInput = trainingDataSequence_sales[startLocation:len(targetDataSequence_sales),:,:]
                trainingBatchTarget = targetDataSequence_sales[startLocation:len(targetDataSequence_sales)]
                
                (_, lsBatch, forecastBatch, forecastBatch_originalScale) = sess.run([minimize_step, loss, forecast, forecast_originalScale], feed_dict={inputSequencePlaceholder: trainingBatchInput, \
                    targetPlaceholder: trainingBatchTarget})
                
                print('got a loss of:', lsBatch)
                print('the forecast of first 5 normalised are:', forecastBatch[0:5])
                print('while the actuals were normalised     :', trainingBatchTarget[0:5])
                print('the forecast of first 5 orignal scale are:', forecastBatch_originalScale[0:5])
                print('while the actuals were original scale     :', (trainingBatchTarget[0:5] * trainingData_sales_range) + trainingData_sales_min)
            
            
            
            totalValidationLoss = 0
            startLocation = 0
            print('starting validation')
            for iter in range(len(validationDataSequence_sales) // BATCH_SIZE):
                validationBatchInput = validationDataSequence_sales[startLocation:startLocation + BATCH_SIZE,:,:]
                validationBatchTarget = validationDataSequence_sales_target[startLocation:startLocation + BATCH_SIZE]
                
                (validationLsBatch, validationForecastBatch, validationForecastBatch_originalScale) = sess.run([loss, forecast, forecast_originalScale], feed_dict={inputSequencePlaceholder: validationBatchInput, \
                    targetPlaceholder: validationBatchTarget})
                
                
                startLocation += BATCH_SIZE
                totalValidationLoss += validationLsBatch
                
                print('first five predictions:', validationForecastBatch[0:5])
                print('first five actuals    :', validationBatchTarget[0:5])
                print('the forecast of first 5 orignal scale are:', validationForecastBatch_originalScale[0:5])
                print('while the actuals were original scale     :', (validationBatchTarget[0:5] * trainingData_sales_range) + trainingData_sales_min)
            
            
            if startLocation < len(validationDataSequence_sales):
                validationBatchInput = validationDataSequence_sales[startLocation:len(validationDataSequence_sales)]
                validationBatchTarget = validationDataSequence_sales_target[startLocation:len(validationDataSequence_sales)]
                
                (validationLsBatch, validationForecastBatch) = sess.run([loss, forecast], feed_dict={inputSequencePlaceholder: validationBatchInput, \
                    targetPlaceholder: validationBatchTarget})
                
                totalValidationLoss += validationLsBatch
            
            
            print('Validation completed after epoch:', e + 1, '. Total validation loss:', totalValidationLoss)
        
        
        print('----------- Saving Model')
        tf.saved_model.simple_save(sess, export_dir=TRAINED_MODEL_PATH, inputs=\
            {'inputSequencePlaceholder': inputSequencePlaceholder, 'targetPlaceholder': targetPlaceholder}, outputs=\
            {'loss': loss, 'forecast_originalScale': forecast_originalScale})
        print('saved model to:', TRAINED_MODEL_PATH)
    
    print('----------- Finis')