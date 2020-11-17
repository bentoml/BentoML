import * as React from 'react'
import Label from './Label';
import {map} from 'lodash';

export interface ILabelDetailSectionProps {
  labels: any;
}

const LabelDetailSection: React.FC<ILabelDetailSectionProps> = (props) => {
  return (
    <p>
      <b>Labels: </b> {map(props.labels, (value, name)=> {
        return (<Label name={name} value={value}/>)
      })}
    </p>
  )
};

export default LabelDetailSection;
