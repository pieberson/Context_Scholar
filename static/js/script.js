document.addEventListener("DOMContentLoaded", () => {
    const overlay = document.getElementById("loadingOverlay");
    document.querySelectorAll("#normalForm, #experimentForm, #normalFormResult, #experimentFormResult")
        .forEach(form => form.addEventListener("submit", () => {
            if (overlay) overlay.style.display = "block";
        }));
}); 

document.addEventListener("DOMContentLoaded", function() {
  const customRangeRadio = document.getElementById("customRange");
  const mostRecentRadio = document.getElementById("mostRecent");
  const dateRangeFields = document.getElementById("customYearFields");

  // Only run this logic if those elements exist on the page
  if (!customRangeRadio || !mostRecentRadio || !dateRangeFields) {
    console.debug("🧠 DEBUG: Date range controls not found — skipping toggleDateRange setup.");
    return; // exit early
  }

  function toggleDateRange() {
    if (customRangeRadio.checked) {
      dateRangeFields.style.display = "block";
    } else {
      dateRangeFields.style.display = "none";
    }
  }

  // Initialize display on page load
  toggleDateRange();

  // Add event listeners
  customRangeRadio.addEventListener("change", toggleDateRange);
  mostRecentRadio.addEventListener("change", toggleDateRange);
}); 

