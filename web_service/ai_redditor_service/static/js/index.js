// $.fn.truncatedText = function() {
//     // Gets the truncated text of an element
//     // Source: https://stackoverflow.com/a/30328736/7614083
//     var o = s = this.text();
//     while (s.length && (this[0].scrollWidth > this.innerWidth())) {
//         s = s.slice(0, -1);
//         this.text(s + "…");
//     }

//     this.text(o);
//     return s;
// }

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

    function togglePostViewVisibility(isPostViewActive) {
        const generateButtonElement = $('#generate-button');
        const postViewElement = $(generateButtonElement.data('post-view-target'));
        const generateViewElement = $(generateButtonElement.data('generate-view-target'));

        if (isPostViewActive) {
            postViewElement.removeClass('d-none');
            generateViewElement.addClass('d-none');
        } else {
            postViewElement.addClass('d-none');
            generateViewElement.removeClass('d-none');
        }
    }

    // Handle changing the view between generated post and generate new post
    $('#generate-button').on('click', function () {
        togglePostViewVisibility(false);
    });

    $('#cancel-generate-button').on('click', function () {
        togglePostViewVisibility(true);
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
