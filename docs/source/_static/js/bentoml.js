function addIcon() {
    const logo = "https://raw.githubusercontent.com/bentoml/BentoML/main/docs/source/_static/js/logo.svg";
    const img = document.createElement("img");
    img.setAttribute("src", logo);
    const div = document.createElement("div");
    div.appendChild(img);
    div.style.textAlign = 'center';
    div.style.maxWidth = '20%';
    div.style.marginLeft= '8rem';
    div.style.paddingTop = '1em';
    div.style.backgroundColor = '#EBECED';

    const scrollDiv = document.querySelector(".wy-side-scroll");
    scrollDiv.prepend(div);
}

function onLoad() {
    addIcon();
}

window.addEventListener("load", onLoad);
