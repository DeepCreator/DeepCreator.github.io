// var seed = 777

const x1_data = [73., 93., 89., 96., 73.]
const x2_data = [80., 88., 91., 98., 66.]
const x3_data = [75., 93., 90., 100., 70.]

const y_data = [152., 185., 180., 196., 142.]

const maxEpoch = 2000
const printInterval = 100

// main function
function main(){
    var x1_train = tf.tensor1d(x1_data)
    var x2_train = tf.tensor1d(x2_data)
    var x3_train = tf.tensor1d(x3_data)
    var y_train = tf.tensor1d(y_data)

    var w1 = tf.variable(
            tf.randomNormal([1],0,1,'float32') 
            , true 
            , 'weight1' 
            , 'float32' 
            )
    var w2 = tf.variable(
            tf.randomNormal([1],0,1,'float32') 
            , true 
            , 'weight2' 
            , 'float32'
            )
    var w3 = tf.variable(
            tf.randomNormal([1],0,1,'float32') 
            , true 
            , 'weight3' 
            , 'float32' 
            )
    var b = tf.variable(
        tf.randomNormal([1],0,1,'float32')
        , true 
        , 'bias' 
        , 'float32'
        )

    var xArray = [x1_train,x2_train,x3_train]
    var wArray = [w1,w2,w3]

    log(`init w1,w2,w3 : ${w1.dataSync()[0]},${w2.dataSync()[0]},${w3.dataSync()[0]}`)
    log(`init b : ${b.dataSync()}`)

    function predict(){
        return tf.tidy(() => {
            return _.chain(_.zip(wArray,xArray)).reduce(
                function(prev,elem){return prev.add(elem[0].mul(elem[1]))} // w1*x + w2*x +w3*x  
                ,b // b
            ).value()
        });
    }

    function loss(pred, label){
        return tf.tidy(() => {
            return pred.sub(label).square().mean();
        });
    }

    const learning_rate=0.00001
    optimizer = tf.train.sgd(learning_rate)


    for (let i = 0; i <= maxEpoch; i++) {
        optimizer.minimize(()=>loss(predict(xArray),y_train));
        if(i%printInterval==0){
            log(`[iter ${i+1}] loss : ${loss(predict(xArray),y_train).dataSync()}`)
            log(`[iter ${i+1}] Prediction:: ${predict(xArray).dataSync()}`)
        }
    }

    // after training
    log(`w1,w2,w3: ${_.chain(wArray).map(x=>x.dataSync()[0]).value()}, b: ${b.dataSync()}`)
}
