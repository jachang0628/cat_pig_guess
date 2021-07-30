// Create Event as well as drawing action
window.addEventListener("load", () => {
    const canvas = document.querySelector('#canvas');
    const context = canvas.getContext('2d');
    canvas.height = 500
    canvas.width = 500

    let painting = false;

    function startpos(e) {
        painting = true;
        draw(e);
    }
    function finpos() {
        painting = false;
        context.beginPath();
    }

    function draw(e) {
        if (!painting) return;
        context.lineWidth = 5;
        context.lineCap = 'round';
        context.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
        context.stroke();
        context.beginPath();
        context.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);


    }
    function stopdraw() {
        painting = false;
        context.beginPath();
    }

    canvas.addEventListener('mousedown', startpos);
    canvas.addEventListener('mouseup', finpos);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseout', stopdraw);

});

// Clear the drawing board
function cleardraw() {
    const canvas = document.querySelector('#canvas');
    const context = canvas.getContext('2d')
    context.clearRect(0, 0, canvas.width, canvas.height);
}

// Save the Drawn Image
function save() {
    const canvas = document.querySelector('#canvas');
    const img_data = canvas.toDataURL()
    const ia = 'hello'
    $.ajax({
        type: "POST",
        url: "/predict",
        data: {
            imageBase64: img_data
        }
    });
}