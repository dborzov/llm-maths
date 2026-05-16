Create a style / design guide and component library to be used across this website. 

We want to build upon pre-existing currently disparate reusable article elements (e.g. The Timeline Widget, Pyplot Executable Code Blocks...) and convert them into a consistent, streamlined and well documented library of reusable components and formatting  leements and convert them into consistent style guide on how to use all of them across the articles:

# What you need to do
The tasks for you:
- Work out through clear, idiomatic library of elements, along with  suggestions on how and when to use each of them within the article, as well as how to actually invoke and utilize them within hugo markdown format when writing articles for this website (using Hugo Shortcodes); 

- Add context-aware AI agent documentation about using this library under `./docs/components/`; rearrange things to move the stuff out of AGENT.md files (AGENTS.md, GEMINI.md, CLAUDE.md); and instead just reference there the head file of that documentation; keep in mind context-preserving concerns so that the details of specific rarely used components do not clog up the context unless they are actually needed;  
- Have a human-facing webpage in `./content/docs` with a style guide, where this component library is explained and described, the components are showcased and we can see what the point is, what a good result of using these components would look like and all that;


# More info

The components we would like to have in our component library:
* Scan the project for the elements we already using that should belong here; markdown formatting elements we already use and how they are rendered into html, at the very least I can see The Timeline Widget and Pyplot Executable Code Blocks, the tech tree graphs headline illustration (which we want to rename to hero images, see more below), Python Snippets(Syntax Highlighting), equations in a standalone block   etc

Others:
1. **Pull Quotes(or Lift-out Quotes):** Large, stylized excerpts of text in a visually distinct panel. We use these to highlight a profound conclusion, a counter-intuitive theorem result etc. Maybe we can have several subtypes with distinct colour schemes and role and tone to serve different roles.

2. **Callout Boxes:** Distinct, visually boxed sections placed alongside the main text.  We can use them for tangent anecdotes, "by the way" sort of notes, or anywhere really anywhere you would want to use a footnote. 

3. **Margin Notes:** Placing citations, corollaries, or minor clarifications in the wide margins next to the relevant text, popularized on the web by Tufte CSS. We have to think how they are rendered visually so that they work both for limited width screens (phones, tablets) and PC screens. This point is that it's  optional content aside from the main narrative flowing.

5. **Crossheads:** Short, descriptive headings used to break up long columns of text. We can think of them as signposts. 
6. 
6. **Hero Image:** Header illustrations we already have should be renamed into "Hero Image" and we should specify what they are more clearly: large, full-width banner image at the top of an article that serves as eye-catching visual candy to help set the mood and visually distinct vibe to the article. 
7.
7. **Figures and plots with Captions:** We want to explicitly document and define how we embed and use 3rd party images with figures and plots and diagrams  explanatory text placed directly below an image or illustration. We want to have html UI where you can get the fullscreen of the image to see the image in details, and make the caption to it rendered nicely, and be able to get back and all that.

8. **Syntax Highlighting Python Snippets:** Distinctly formatted, often dark-themed background blocks used specifically for code snippets or command-line inputs. This provides immediate visual differentiation between standard prose, mathematical equations, and executable code.

10. **Interactive Data Visualization Elements (Infographics):** Here is what we want this to look like and to do. Imagine we are making some technical argument and to prove it we quote some  table of numbers (say, the size of KV cache for distinct context lengths for different LLMs). Instead of representing this as static table, we  rework this into a little snippet of embedded  interactive HTML element where the user is given some interactive toggles (nothing too fancy just some layouts of panes & panels of different colours, buttons and toggle buttons, dropdown selectors, tabs for them, check), and so on. The point is to have option to add some interactivity so that instead of the wall of numbers static table, the human/reader gets to play around and tinker with the different options and see the data interactively. They can see and compare the data points from the table in a way that would illustrate and convey the technical point /argument this data is saying in much more fun way. We must be careful to keeping the scope of what sort of these mini-interactive visualization tools we would allow and specify so that we dont spend too much time and effort on that.

# How to use Hugo shortcodes for that

I am not sure how to make and specify all this technically, wold we be able to allow defining all these components using Hugo shortcodes?

In the Hugo ecosystem, the idiomatic term for these reusable layout components is **Shortcodes**.

Shortcodes are custom HTML templates you define once and can invoke anywhere inside your Markdown files using a specific bracket syntax. This keeps your Markdown clean and separates your content from your visual presentation.

Here is how to build and implement a library of shortcodes for your blog:

1. **Create the Shortcode Directory**
All custom components must be stored as individual `.html` files inside the `layouts/shortcodes/` directory at the root of your Hugo project (or within your theme's layout folder). The name of the file becomes the name of the shortcode.
2. **Build a Pull Quote Shortcode**
Create a file at `layouts/shortcodes/pullquote.html`. You use Go Template syntax to accept arguments from the Markdown file.
```html
<aside class="pull-quote">
    <span class="quote-text">"{{ .Get "text" }}"</span>
    {{ with .Get "author" }}
        <span class="quote-author">— {{ . }}</span>
    {{ end }}
</aside>

```


3. **Call the Pull Quote in Markdown**
Use the double curly brace and angle bracket syntax to pass parameters to the shortcode in any `.md` content file.

```markdown
   {{< pullquote text="The most dangerous phrase in the language is, 'We've always done it this way.'" author="Grace Hopper" >}}

```

4. **Build a Sidebar / Callout Box Shortcode**
Create a file at `layouts/shortcodes/callout.html`. To allow complex content (like multiple paragraphs, lists, or math equations) inside the component, you use the `.Inner` variable.

```html
   <div class="callout-box">
       {{ if .Get "title" }}
           <strong class="callout-title">{{ .Get "title" }}</strong>
       {{ end }}
       <div class="callout-content">
           {{ .Inner }}
       </div>
   </div>

```

5. **Call the Callout in Markdown**
When using `.Inner`, you must use an opening and closing tag. Crucially, to ensure Hugo processes standard Markdown *inside* the shortcode, use the `%` delimiter instead of `<`.

```markdown
   {{% callout title="Theorem 1.2 Definition" %}}
   This is the inner content where you can use standard Markdown, like **bolding** or bullet points, to explain the mathematical proof.
   {{% /callout %}}

```

6. **Centralize Your Styling (CSS)**
Notice the HTML above only assigns CSS classes (like `class="pull-quote"`). You should define the actual visual candy (colors, borders, large fonts, drop shadows) in your project's main CSS file. This allows you to update the visual design globally without editing the individual shortcode files.

How are you currently handling CSS and math rendering (such as KaTeX or MathJax) in your Hugo setup so we can ensure these shortcodes accommodate your equations properly?

```

```
