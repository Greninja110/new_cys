/**
 * Main JavaScript file for the NIDS application
 */

document.addEventListener('DOMContentLoaded', function() {
    // File upload handling
    const fileInput = document.getElementById('file');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const removeFile = document.getElementById('remove-file');
    const uploadForm = document.getElementById('upload-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = document.getElementById('btn-text');
    const loadingSpinner = document.getElementById('loading-spinner');

    // Handle file selection
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                const file = this.files[0];
                fileName.textContent = file.name;
                fileSize.textContent = formatFileSize(file.size);
                fileInfo.classList.remove('d-none');
            } else {
                fileInfo.classList.add('d-none');
            }
        });
    }

    // Handle file removal
    if (removeFile) {
        removeFile.addEventListener('click', function() {
            fileInput.value = '';
            fileInfo.classList.add('d-none');
        });
    }

    // Handle form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', function() {
            btnText.textContent = 'Analyzing...';
            loadingSpinner.classList.remove('d-none');
            analyzeBtn.setAttribute('disabled', true);
        });
    }

    // Format file size
    function formatFileSize(bytes) {
        if (bytes < 1024) {
            return bytes + ' bytes';
        } else if (bytes < 1024 * 1024) {
            return (bytes / 1024).toFixed(2) + ' KB';
        } else {
            return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
        }
    }

    // Tooltips initialization
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Charts resizing
    function resizeCharts() {
        const charts = document.querySelectorAll('[id^="chart"]');
        charts.forEach(chart => {
            if (chart && typeof Plotly !== 'undefined') {
                Plotly.relayout(chart.id, {
                    'width': chart.offsetWidth
                });
            }
        });
    }

    // Handle window resize for charts
    window.addEventListener('resize', function() {
        // Debounce resize event
        clearTimeout(window.resizeTimeout);
        window.resizeTimeout = setTimeout(resizeCharts, 250);
    });

    // Flash message auto-dismiss
    const flashMessages = document.querySelectorAll('.alert-dismissible');
    flashMessages.forEach(message => {
        setTimeout(() => {
            const closeButton = message.querySelector('.btn-close');
            if (closeButton) {
                closeButton.click();
            }
        }, 5000); // Auto-dismiss after 5 seconds
    });

    // Accordion state persistence
    const accordions = document.querySelectorAll('.accordion');
    accordions.forEach(accordion => {
        const accordionId = accordion.id;
        const activeItem = localStorage.getItem(`accordion_${accordionId}`);
        
        if (activeItem) {
            const item = document.getElementById(activeItem);
            if (item) {
                const bsCollapse = new bootstrap.Collapse(item, {
                    toggle: false
                });
                bsCollapse.show();
            }
        }

        // Save accordion state on change
        accordion.addEventListener('shown.bs.collapse', function(e) {
            localStorage.setItem(`accordion_${accordionId}`, e.target.id);
        });
    });

    // Tab state persistence
    const tabEls = document.querySelectorAll('button[data-bs-toggle="tab"]');
    tabEls.forEach(tabEl => {
        tabEl.addEventListener('shown.bs.tab', function (e) {
            const tabId = e.target.id;
            const tabGroup = e.target.closest('.nav-tabs').id;
            localStorage.setItem(`tab_${tabGroup}`, tabId);
        });

        // Restore active tab
        const tabGroup = tabEl.closest('.nav-tabs').id;
        const activeTab = localStorage.getItem(`tab_${tabGroup}`);
        if (activeTab && activeTab === tabEl.id) {
            const tab = new bootstrap.Tab(tabEl);
            tab.show();
        }
    });

    // API status check
    function checkApiStatus() {
        const statusIndicator = document.getElementById('api-status');
        if (statusIndicator) {
            fetch('/api/status')
                .then(response => {
                    if (response.ok) {
                        statusIndicator.innerHTML = '<span class="badge bg-success">Online</span>';
                    } else {
                        statusIndicator.innerHTML = '<span class="badge bg-warning">Degraded</span>';
                    }
                })
                .catch(error => {
                    statusIndicator.innerHTML = '<span class="badge bg-danger">Offline</span>';
                    console.error('API Status Check Error:', error);
                });
        }
    }

    // Check API status on load if indicator exists
    if (document.getElementById('api-status')) {
        checkApiStatus();
        // Periodically check status
        setInterval(checkApiStatus, 60000); // Check every minute
    }
});