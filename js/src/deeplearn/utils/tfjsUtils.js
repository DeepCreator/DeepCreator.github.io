// dependency on tensorflow.js ^0.9.0 and underscore ^1.9.0

/**
 * 
 * @param {object} tensor
 * @returns {object} stacked array
 */
function ToStackedArray(tensor,digit=undefined){
    if(digit && typeof digit != "number"){
        digit = Number(digit)
        console.log(digit)
    }
    // console.log(_.last(tensor.shape,tensor.shape.length-1))
    return _.chain(_.last(tensor.shape,tensor.shape.length-1)).reduceRight(
        function(accumulator,currentValue){
            return _.chunk(accumulator,currentValue)
        }
        , digit && digit >= 0 ? tensor.dataSync().map(x=>x.toFixed(digit)) : tensor.dataSync()
    ).value()
}


/**
 * courtesy by engelen
 * Retrieve the array key corresponding to the largest element in the array.
 *
 * @param {Array.<number>} array Input array
 * @return {number} Index of array element with largest value
 */
function argMax(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

/**
 * tf.onehot has bug for now. 2018-05-02
 * @param {tensor} tensor  assume 2D or 1D tensor
 */
function toOneHot(tensor,nb_classes){
    return tf.tensor2d(
        _.map(ToStackedArray(tensor),(row)=>{
            let newRow = Array(nb_classes).fill(0)
            if(typeof row === "object"){
                newRow[row[0]] = 1
            }else{
                newRow[row] = 1
            }
            // console.log("row",row)
            // console.log("newRow",newRow)
            return newRow
        })
    )
    // return tf.stack(_.map(tensor,(label)=>{
    //     tf.buffer(tensor.,'int32').set(1,label).toTensor()
    // }))
}
  