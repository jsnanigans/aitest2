Syntax Guide

Slidev's slides are written as Markdown files, which are called Slidev Markdowns. A presentation has a Slidev Markdown as its entry, which is ./slides.md by default, but you can change it by passing the file path as an argument to the CLI commands.

In a Slidev Markdown, not only the basic Markdown features can be used as usual, Slidev also provides additional features to enhance your slides. This section covers the syntax introduced by Slidev. Please make sure you know the basic Markdown syntax before reading this guide.
Slide Separators

Use --- padded with a new line to separate your slides.

# Title

Hello, **Slidev**!

---

# Slide 2

Use code blocks for highlighting:

```ts
console.log('Hello, World!')
```

---

# Slide 3

Use UnoCSS classes and Vue components to style and enrich your slides:

<div class="p-3">
  <Tweet id="..." />
</div>

Frontmatter & Headmatter

At the beginning of each slide, you can add an optional frontmatter to configure the slide. The first frontmatter block is called headmatter and can configure the whole slide deck. The rest are frontmatters for individual slides. Texts in the headmatter or the frontmatter should be an object in YAML format. For example:

---
theme: seriph
title: Welcome to Slidev
---

# Slide 1

The frontmatter of this slide is also the headmatter

---
layout: center
background: /background-1.png
class: text-white
---

# Slide 2

A page with the layout `center` and a background image

---

# Slide 3

A page without frontmatter

---
src: ./pages/4.md  # This slide only contains a frontmatter
---

---

# Slide 5

Configurations you can set are described in the Slides deck configurations and Per slide configurations sections.
