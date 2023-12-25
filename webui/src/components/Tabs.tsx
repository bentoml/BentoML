import { useStyletron } from 'baseui'
import type { TabsProps } from 'baseui/tabs'
import { Tabs as BaseUITabs, Tab } from 'baseui/tabs'

interface Props extends Omit<TabsProps, 'children'> {
  children: any // TODO: We are not sure how to deal with this type definition for now, We will look at it later.
}

function Tabs(props: Props) {
  const [css, theme] = useStyletron()

  return (
    <BaseUITabs
      {...props}
      overrides={{
        TabBar: {
          props: {
            className: css({
              backgroundColor: 'transparent!important',
              paddingLeft: '0!important',
              paddingRight: '0!important',
              borderBottomWidth: '1px',
              borderBottomStyle: 'solid',
              borderBottomColor: `${theme.colors.borderOpaque}`,
            }),
          },
        },
        TabContent: {
          props: {
            className: css({
              paddingLeft: '0!important',
              paddingRight: '0!important',
            }),
          },
        },
      }}
    />
  )
}

export { Tabs, Tab }
export default Tabs
