document.addEventListener("DOMContentLoaded", function() {
    const normalForm = document.getElementById("normalForm");
    const experimentForm = document.getElementById("experimentForm");
    const loadingOverlay = document.getElementById("loadingOverlay");

    [normalForm, experimentForm].forEach(form => {
        form.addEventListener("submit", function() {
            loadingOverlay.style.display = "block";
        });
    });
});
