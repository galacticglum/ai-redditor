import React, { Component } from 'react';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form';

import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Tooltip from 'react-bootstrap/Tooltip';

import './ConfigPage.css';

const WithTooltip = (props) => (
    <OverlayTrigger
        placement={props.placement}
        delay={{ show: 500 }}
        overlay={
            <Tooltip>
                {props.text}                                                  
            </Tooltip>
        }
    >
        {props.children}
    </OverlayTrigger>
)

WithTooltip.defaultProps = {
    placement: "top"
}

export default class ConfigPage extends Component {
    constructor(props) {
        super(props);
        this.state = {
            recordTypes: {
                tifu: false,
                wp: false,
                phc: true
            }
        };
    }

    recordTypeToggleChange = (event) => {
        this.setState({
            recordTypes: {
                ...this.state.recordTypes,
                [event.target.id]: event.target.checked
            }
        });
    }

    render() {
        return (
            <Container>
                <Row>
                    <Col sm="12" md="8" lg="6" className="mx-auto">
                        <div className="settings-panel">
                            <h4 className="text-center mt-2 font-weight-bold">settings</h4>
                            <Form className="mt-4">
                                <Form.Group>
                                    <Form.Label className="font-weight-bold">types of posts?</Form.Label>
                                    <div>
                                        <Form.Check id="tifu" inline custom className="record-type-checkbox">
                                            <Form.Check.Input type="checkbox"
                                                onChange={this.recordTypeToggleChange}
                                                checked={this.state.recordTypes['tifu']} />
                                            <WithTooltip text="today i fucked up">
                                                <Form.Check.Label>tifu</Form.Check.Label>
                                            </WithTooltip>
                                        </Form.Check>
                                        <Form.Check id="wp" inline custom className="record-type-checkbox">
                                            <Form.Check.Input type="checkbox"
                                                onChange={this.recordTypeToggleChange}
                                                checked={this.state.recordTypes['wp']} />
                                            <WithTooltip checked={this.state.recordTypes['wp']} text="writing prompts">
                                                <Form.Check.Label>wp</Form.Check.Label>
                                            </WithTooltip>
                                        </Form.Check>
                                        <Form.Check id="phc" inline custom className="record-type-checkbox">
                                            <Form.Check.Input type="checkbox"
                                                onChange={this.recordTypeToggleChange}
                                                checked={this.state.recordTypes['phc']} />
                                            <WithTooltip text="pornhub comments">
                                                <Form.Check.Label>phc</Form.Check.Label>
                                            </WithTooltip>
                                        </Form.Check>  
                                    </div>
                                </Form.Group>
                                <Form.Group className="mt-4">
                                    <Form.Label className="font-weight-bold">max guessing time (seconds)</Form.Label>
                                    <Form.Control type="range" custom />
                                </Form.Group>
                            </Form>
                        </div>
                    </Col>
                </Row>
            </Container>
        )
    }
}
