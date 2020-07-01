import {MnistData} from './data.js';

let canvas;
let ctx;
let rawImage;
let pos = {x:0, y:0};

let model;


function erase() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 280, 280);
}

function getModel() {
    model = tf.sequential();
    model.add(tf.layers.conv2d({inputShape : [28, 28, 1], kernelSize: 3, filters: 8, activation: 'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
    model.add(tf.layers.conv2d({kernelSize: 3, filters: 16, activation: 'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({units: 128, activation: 'relu'}));
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
    model.compile({ optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy']})

    return model;
}

async function train(model, data) {
    
    // Set up display training progress
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {name: 'Model Training', styles: {height: '640px'}}
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;

    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
            d.labels
        ]
    })


    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
            d.labels
        ]
    })

    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 20,
        shuffle: true,
        callbacks: fitCallbacks
    })

}


function save() {
    console.log('Saving...');

    raw = tf.browser.fromPixels(rawImage, 1);
    resize = tf.image.resizeBilinear(raw, [20, 20]);
    tensor = resize.expandDims(0);
    tensor.print();
}

function setPosition(e) {
    pos.x = e.clientX - canvas.getBoundingClientRect().left
    pos.y = e.clientY - canvas.getBoundingClientRect().top
    console.log(pos)

}

function draw(e) {
    if(e.buttons != 1) return;
    console.log('Drawing');
    ctx.beginPath();
    ctx.lineWidth = 24;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';
    ctx.moveTo(pos.x, pos.y)
    setPosition(e);
    ctx.lineTo(pos.x, pos.y)
    ctx.stroke();

    rawImage.src = canvas.toDataURL('image/png');
}

async function init() {
    const eraseButton = document.getElementById('erase-button')
    const saveButton = document.getElementById('save-button')

    eraseButton.addEventListener('click', erase)
    saveButton.addEventListener('click', save)

    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 280, 280);

    canvas.addEventListener('mousedown', setPosition)
    canvas.addEventListener('mousemove', draw)

    rawImage = document.getElementById('canvas-img')

    const data = new MnistData();
    await data.load();
    const model = getModel();
    tfvis.show.modelSummary({name: 'Model Architecture', }, model)
    await train(model, data);
    alert("Training is done, try classifying your handwriting!");
    
    
}

document.addEventListener('DOMContentLoaded', init)