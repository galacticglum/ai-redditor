$.fn.truncate = function (lines) {
    lines = typeof lines !== 'undefined' ? lines : 1;
    var lineHeight = parseInt(this.css('line-height'));
    const doTruncate = this.height() > lines * lineHeight;
    if (doTruncate) {
        var words = this.html().split(' ');
        var str = "";
        var prevstr = "";
        this.text("");
        for (var i = 0; i < words.length; i++) {
            if (this.height() > lines * lineHeight) {
                this.html(prevstr.trim() + '&hellip;');
                break;
            }
            prevstr = str;
            str += words[i] + ' ';
            this.html(str.trim() + '&hellip;');
        }
        if (this.height() > lines * lineHeight) {
            this.html(prevstr.trim() + '&hellip;');
        }
    }

    return this, doTruncate;
}

function unescapeHtml(safe) {
    return safe.replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        .replace(/&#039;/g, "'");
}

const Views = {
    POST: 'post',
    GENERATE: 'generate',
    LOADING: 'loading',
    ERROR: 'error'
}

function getViewElement(x) { 
    return $(`#${x}-view`); 
}

const ViewValues = Object.values(Views);
function toggleView(view) {
    if (!ViewValues.includes(view)) return;
    ViewValues.filter(x => x != view).forEach(x => getViewElement(x).addClass('d-none'));
    getViewElement(view).removeClass('d-none');
}

$(document).ready(function () {
    $('body').show();

    // Redacted text effect
    $('.redacted').hover(function () {
        $(this).toggleClass('redacted-remove-hover');
    });

    // Check if collapse-toggle should be active (is the source text overflowing?)
    const collapseToggleElement = $('#collapse-toggle');
    if (collapseToggleElement.length) {
        const postBodyTextElement = $(collapseToggleElement.data('expand-target'));

        // Truncate content
        const postContent = unescapeHtml($('#post-content').html());
        postBodyTextElement.html(postContent);

        const isReadmoreVisible = postBodyTextElement.truncate(3);
        collapseToggleElement.toggle(isReadmoreVisible);

        // Post body expand ("read more") button
        collapseToggleElement.on('click', function () {
            // Toggling truncation works by changing the display CSS attribute.
            var expanded = $(this).data('expanded');
            if (expanded === undefined) {
                expanded = false;
            }

            expanded = !expanded;
            const expandTargetElement = $($(this).data('expand-target'));
            const expandLabelElement = $($(this).data('expand-label-target'));
            expandTargetElement.html(postContent);
            if (expanded) {
                expandLabelElement.text('Read less');
            } else {
                expandTargetElement.html(postContent);
                expandTargetElement.truncate(3);
                expandLabelElement.text('Read more');
            }

            // Toggle padding on the footer
            $('.mastfoot').toggleClass('pb-3');
            collapseToggleElement.data('expanded', expanded);
        });
    }

    // Handle changing the view between generated post and generate new post
    $('#generate-button').on('click', function () {
        toggleView(Views.GENERATE);
    });

    $('#cancel-generate-button, #close-error-button').on('click', function () {
        toggleView(Views.POST);
    });

    // Initialize copy to clipboard action for permalink button
    new ClipboardJS('#permalink');

    // Enable tooltips everywhere
    $('[data-toggle="tooltip"]').tooltip();

    // Disable permalink link action
    $('#permalink').on('click', function (e) {
        const element = $(this);
        element.tooltip('show');
        setTimeout(function () {
            element.tooltip('hide');
        }, 500);

        e.preventDefault();
    });
});
