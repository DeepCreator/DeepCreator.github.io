var data = [[1,0,1,0,1,1,1,0,0,0,0,0,1,0],
            [1,1,1,1,1,1,1,0,0,0,0,0,1,0],
            [1,1,1,0,1,1,1,0,1,0,0,0,1,0],
            [1,0,1,1,1,1,1,1,0,0,0,0,1,0],
            [1,1,1,1,1,1,1,0,0,0,0,0,1,1],
            [0,0,1,0,0,1,0,0,1,0,1,1,1,0],
            [0,0,0,0,0,0,1,1,1,0,1,1,1,0],
            [0,0,0,0,0,1,1,1,0,1,0,1,1,0],
            [0,0,1,0,1,0,1,1,1,1,0,1,1,1],
            [0,0,0,0,0,0,1,1,1,1,1,1,1,1],
            [1,0,1,0,0,1,1,1,1,1,0,0,1,0]
           ];

var result = [23,12,23,23,45,70,123,73,146,158,64];

var knn = new ml.KNN({
    data : data,
    result : result
});

var y = knn.predict({
    x : [0,0,0,0,0,0,0,1,1,1,1,1,1,1],
    k : 3,

    weightf : {type : 'gaussian', sigma : 10.0},
    // default : {type : 'gaussian', sigma : 10.0}
    // {type : 'none'}. weight == 1
    // Or you can use your own weight f
    // weightf : function(distance) {return 1./distance}

    distance : {type : 'euclidean'}
    // default : {type : 'euclidean'}
    // {type : 'pearson'}
    // Or you can use your own distance function
    // distance : function(vecx, vecy) {return Math.abs(dot(vecx,vecy));}
});

console.log(y);