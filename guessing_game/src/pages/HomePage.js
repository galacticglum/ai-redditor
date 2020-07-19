import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import Container from 'react-bootstrap/Container';
import Button from 'react-bootstrap/Button';
import './HomePage.css';

export default class HomePage extends Component {
    render() {
        return (
            <Container className="d-flex flex-column align-items-center p-100">
                <div className="text-center title title-responsive-size">
                    ai or human?
                </div> 
                <div className="pt-5 explanation-text text-center">
                    <p>
                        doesn't really need much of an explanation; just guess if the post was written by an <a className="text-link"
                        href="https://www.youtube.com/watch?v=dQw4w9WgXcQ">artificial intelligence</a> or by an actual <a className="text-link"
                        href="https://en.wikipedia.org/wiki/Human">human</a>.
                    </p>
                </div>

                <Link to="/play">
                    <Button size="lg" className="my-5 play-btn">
                        play
                    </Button>
                </Link>
                
                <div className="pt-4 explanation-text text-center">
                    <p>
                        built by <a className="text-link" href="https://github.com/galacticglum">shon verch</a> (<a 
                        class="text-link" href="https://twitter.com/galacticglum">@galacticglum</a>) using a
                        finetuned generative language model (<a class="text-link" href="https://openai.com/blog/better-language-models/">gpt2</a>).
                    </p>
                </div>

                <a className="make-your-own-btn text-link" href={process.env.REACT_APP_TPDNE_BASE_URL}>
                    <span role="img" aria-label="memo">&#128221;</span> generate your own post
                </a>
            </Container>
        )
    }
}
