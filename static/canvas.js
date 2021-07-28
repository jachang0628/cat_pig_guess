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
    canvas.addEventListener('mouseout', stopdraw)

});

function cleardraw() {
    const canvas = document.querySelector('#canvas');
    const context = canvas.getContext('2d')
    context.clearRect(0, 0, canvas.width, canvas.height);
}