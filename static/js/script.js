document.addEventListener("DOMContentLoaded", () => {
    const overlay = document.getElementById("loadingOverlay");
    document.querySelectorAll("#normalForm, #experimentForm, #normalFormResult, #experimentFormResult")
        .forEach(form => form.addEventListener("submit", () => {
            if (overlay) overlay.style.display = "block";
        }));
<<<<<<< HEAD
});
=======
});
>>>>>>> origin/jho_branch
